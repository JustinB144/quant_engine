# Quant Engine System Intent & Component Audit (Source-Based)

## What This Document Is

This is an intention/component audit of the `quant_engine` system written from actual source code review (not guessed behavior).
It is designed for two audiences:

- you (to understand how the system is supposed to operate end-to-end), and
- another LLM (to ingest the architecture, invariants, and component responsibilities before modifying the system).

## Audit Method (How This Was Built)

This report was built from:

- direct source reads of the core execution paths (data loading, features, regime detection, modeling, prediction, backtest, risk, autopilot, Kalshi pipeline, Dash UI pages/components),
- a fresh AST-based parse of the current repo after cleanup to enumerate modules/classes/functions, and
- a Dash UI callback/page map generated from the actual Dash page source.

Important: this is intentionally **intention-oriented** (what each component is for and why it exists), not a code walkthrough.

## Current-State Summary (Post-Cleanup)

- Active UI: `dash_ui` (Dash) launched by `run_dash.py`.
- Legacy UI: `ui/` + `run_dashboard.py` removed in this cleanup.
- Root loose docs/notes were moved under `docs/` (`guides/`, `reports/`, `plans/`, `notes/`).
- Exact duplicate-content files found: only one benign pair of minimal package init files.
- Core system architecture remains intact (data -> features -> regimes -> model -> backtest/risk -> autopilot -> Dash monitoring/control).

## Repository Component Coverage (Current Source Parse)

| Package | Modules | Classes | Top-level Functions | LOC |
|---|---:|---:|---:|---:|
| `(root)` | 12 | 0 | 23 | 2,561 |
| `autopilot` | 6 | 9 | 0 | 1,770 |
| `backtest` | 6 | 15 | 16 | 3,372 |
| `dash_ui` | 26 | 2 | 123 | 9,726 |
| `data` | 10 | 13 | 55 | 5,164 |
| `features` | 9 | 3 | 43 | 3,047 |
| `indicators` | 2 | 92 | 2 | 2,692 |
| `kalshi` | 25 | 34 | 66 | 5,947 |
| `models` | 13 | 26 | 7 | 4,397 |
| `regime` | 4 | 5 | 5 | 1,019 |
| `risk` | 11 | 14 | 14 | 2,742 |
| `tests` | 20 | 29 | 9 | 2,572 |
| `utils` | 2 | 3 | 1 | 437 |

## System Context and Consensus (Cross-Module Contracts)

This system is not just a collection of scripts; it is held together by a set of implicit/explicit contracts. These are the most important ones.

### 1. Identity Consensus: PERMNO-First (Not Ticker-First)

Why it exists:
- Tickers are unstable across time (renames, delistings, symbol reuse).
- Backtests and autopilot decisions must stay tied to the same economic entity across history.

What enforces it:
- `data.loader` and `data.wrds_provider` build/normalize panels around CRSP/WRDS identities.
- `backtest.engine.Backtester` includes runtime assertions for PERMNO-keyed inputs when strict mode is enabled.
- `autopilot.engine.AutopilotEngine` and `autopilot.paper_trader.PaperTrader` support PERMNO-first keys with legacy ticker fallback.

What it means operationally:
- The system treats `ticker` as display/compatibility metadata when possible, and `permno` as the authoritative join key for research and execution simulation.

### 2. Time Consensus: Strict As-Of / No-Leakage Joins

Why it exists:
- The system is research-heavy; leakage is the fastest way to generate false performance.
- Event strategies (Kalshi/macro) are especially vulnerable to timestamp mistakes.

What enforces it:
- `kalshi.events.asof_join(...)` uses strict backward joins.
- `kalshi.events._ensure_asof_before_release(...)` explicitly raises if `asof_ts >= release_ts`.
- `models.trainer` uses panel-safe splits and purged CV logic.
- `kalshi.walkforward.run_event_walkforward(...)` adds purge/embargo on event splits.
- Tests in `kalshi/tests/test_no_leakage.py`, `kalshi/tests/test_leakage.py`, `kalshi/tests/test_walkforward_purge.py` encode these invariants.

### 3. Data Quality Consensus: Prefer Trusted Data, Record Provenance, Fail Softly

Why it exists:
- Market data and event data are messy/incomplete; the system must keep running without silently fabricating certainty.

What enforces it:
- `data.loader` prioritizes trusted cache/WRDS paths, validates quality, and falls back to less authoritative sources only when needed.
- `data.local_cache` stores metadata sidecars (provenance, freshness, source hints).
- `data.quality` and `kalshi.quality` expose quality scores and hard gates so low-quality data can be filtered or down-weighted.
- `kalshi.storage` persists ingestion logs, coverage diagnostics, and daily health reports.

### 4. Regime Consensus: Regime Is a First-Class Context Signal (Not Just a Label)

Why it exists:
- Strategy behavior, position sizing, and prediction trust vary across market states.

What enforces it:
- `regime.detector` produces regime label, confidence, posterior probabilities, duration, and optional transition probability.
- `regime.correlation` adds a second regime dimension (cross-asset correlation spike behavior).
- `models.predictor` blends global and regime-specific experts and can suppress outputs under weak regime confidence.
- `backtest.engine` and `autopilot.paper_trader` use regime-aware gating/sizing (including risk multipliers).

### 5. Promotion Consensus: Performance Alone Is Not Enough

Why it exists:
- The system distinguishes “good backtest” from “deployable strategy.”

What enforces it:
- `autopilot.promotion_gate.PromotionGate` requires trade count, win rate, Sharpe, drawdown, and annual return thresholds.
- Advanced contract checks include DSR significance, PBO, capacity, walk-forward correlation, IS/OOS gap, and regime robustness.
- Kalshi event strategies extend the same promotion logic with event-specific risk checks (worst event loss, surprise hit rate, event regime stability).

## End-to-End System Flows (What the System Is Trying to Do)

## 1. Model Development Flow (Train -> Predict -> Backtest)

Intent:
- Turn historical market data into a predictive signal pipeline with realistic evaluation.

Flow:
1. `run_train.py`
   - loads historical price/fundamental/options inputs via `data.loader`,
   - builds feature panels via `features.pipeline`,
   - builds regime features via `regime.detector` (+ correlation regime features where enabled),
   - trains ensemble/regime-aware models via `models.trainer`,
   - saves versioned artifacts and metadata (`models.versioning`, trainer artifact writer).
2. `run_predict.py`
   - loads the selected model version/champion,
   - rebuilds latest features/regimes,
   - runs `models.predictor` to produce predicted returns + confidence + weights.
3. `run_backtest.py`
   - feeds predictions + OHLCV into `backtest.engine.Backtester`,
   - applies execution/cost/risk logic,
   - computes performance + validation metrics.

Why these pieces are separated:
- You can retrain, predict, and backtest independently.
- It reduces coupling between research, inference, and execution simulation.
- It supports the autopilot pipeline (which reuses the same building blocks).

## 2. Continuous Improvement Flow (Retrain Governance)

Intent:
- Avoid ad-hoc retraining; retrain only when triggers/governance say it is warranted.

Main components:
- `run_retrain.py`: orchestrates retraining checks and retrain runs.
- `models.retrain_trigger`: detects drift/performance decay or schedule-based retraining conditions.
- `models.governance`: defines rules around champion selection / promotion discipline.
- `models.versioning`: manages model versions, registry state, and champion resolution.

Why it exists:
- Prevents performance chasing and accidental replacement of stable production models.

## 3. Autopilot Flow (Discovery -> Validation -> Promotion -> Paper Trading)

Intent:
- Search execution parameter variants around a baseline predictive model, then promote only robust variants.

Main stages in `autopilot.engine.AutopilotEngine.run_cycle()`:
- Load data and assert identity conventions (PERMNO-first expectations).
- Ensure predictor exists:
  - can train baseline model on earliest 80% dates (strict OOS discipline),
  - or fall back to `HeuristicPredictor` if artifacts/dependencies are unavailable.
- Build universe predictions (latest and/or walk-forward OOS predictions).
- Generate strategy candidates using `autopilot.strategy_discovery.StrategyDiscovery`.
- Evaluate candidates with backtesting + advanced validation metrics.
- Apply `autopilot.promotion_gate.PromotionGate` to determine pass/fail.
- Persist active promoted strategies via `autopilot.registry.StrategyRegistry`.
- Run paper execution via `autopilot.paper_trader.PaperTrader`.

Why this matters:
- It upgrades the system from “single strategy research script” to a controlled strategy lifecycle manager.

## 4. Kalshi Event-Market Flow (Ingest -> Distributions -> Event Features -> Walk-Forward -> Promotion)

Intent:
- Treat Kalshi contract quotes as an event-time information surface and build leak-safe event strategies around it.

Main stages:
1. Ingestion (`kalshi.client`, `kalshi.provider`, `kalshi.storage`)
   - signed API access,
   - rate limiting + retries,
   - live/historical endpoint routing,
   - persisted market/contract/quote snapshots + ingestion provenance.
2. Distribution building (`kalshi.distribution`)
   - converts contract quotes into market-level probability distribution snapshots,
   - repairs monotonic threshold curves with isotonic projections when needed,
   - computes moments/tails/entropy + quality metadata + distance features.
3. Event feature panel (`kalshi.events`)
   - builds pre-release as-of snapshot grids,
   - maps macro events to markets,
   - joins Kalshi distribution snapshots backward in time,
   - adds drift/collapse speed features and optional cross-market disagreement features.
4. Event walk-forward (`kalshi.walkforward`)
   - event-level purged/embargoed folds,
   - nested alpha selection for ridge regression,
   - OOS correlation/return metrics,
   - advanced contract metrics (DSR, Monte Carlo, bootstrap, capacity, event-type stability).
5. Event promotion (`kalshi.promotion`, `autopilot.promotion_gate`)
   - wraps event returns into a backtest-like object,
   - routes through the same promotion discipline as standard strategies, plus event-specific checks.

Why the Kalshi vertical is isolated:
- It has very different data semantics (contracts/quotes/distributions/event timestamps) than the equity prediction stack.
- Isolation reduces contamination of the core equity pipeline while still allowing shared validation/promotion standards.

## Subsystem Breakdown (Intention/Components/Why They Exist)

## Entry Points and Orchestration

| File | Intent |
|---|---|
| `run_train.py` | Train and persist model artifacts for a horizon using the feature pipeline + trainer. |
| `run_predict.py` | Load a trained version and produce forward-return predictions with confidence/regime overlays. |
| `run_backtest.py` | Generate predictions and run the backtester (simple or risk-managed modes) with validation outputs. |
| `run_retrain.py` | Evaluate retraining triggers/governance and run controlled retraining workflows. |
| `run_autopilot.py` | Run the strategy autopilot cycle: baseline model, discovery, validation, promotion, paper trading. |
| `run_kalshi_event_pipeline.py` | Run the Kalshi event-market ingestion/feature/walk-forward/promotion pipeline. |
| `run_dash.py` | Launch the current Dash UI (`dash_ui`). |
| `run_wrds_daily_refresh.py` | Operational WRDS refresh orchestration and cache maintenance. |

### Why the root-level run scripts exist

- They provide stable operational commands for recurring workflows.
- They separate operator actions (train, backtest, autopilot, refresh, UI launch) from reusable package logic.
- They keep package modules importable/testable while preserving CLI usability.

## Data Layer (`data/`)

### Core intent
- Normalize acquisition across multiple providers (WRDS, local cache, yfinance fallback) while preserving point-in-time correctness and provenance.

### Key components and why they exist

- `data.loader`
  - Central orchestration for fetching symbol/universe panels.
  - Exists so train/predict/backtest/autopilot all use the same cache/quality/survivorship behavior instead of duplicating fetch logic.
  - Implements the trust/fallback order (trusted cache -> WRDS -> cache fallback -> yfinance fallback).
  - Integrates OptionMetrics-style features when available.

- `data.local_cache`
  - Disk cache abstraction for OHLCV and metadata sidecars.
  - Exists to make expensive data acquisition reproducible and fast.
  - Stores metadata so downstream code can reason about freshness/source/quality, not just raw prices.

- `data.wrds_provider`
  - Primary institutional-grade source adapter.
  - Exists to pull CRSP/Compustat/OptionMetrics/TAQ-derived data with consistent identifiers and point-in-time semantics.
  - Includes delisting return handling and PERMNO->ticker mapping utilities.

- `data.survivorship`
  - Point-in-time universe membership storage/filtering.
  - Exists to prevent survivorship bias when constructing historical panels (especially S&P 500 and other dynamic universes).

- `data.provider_registry`
  - Provider registration/selection layer.
  - Exists to decouple data loading orchestration from concrete provider implementations.

- `data.quality`
  - Quality checks/scoring utilities.
  - Exists so low-quality data can be gated, flagged, or handled explicitly instead of silently used.

## Feature Engineering Layer (`features/`)

### Core intent
- Build a rich feature panel from market, cross-asset, macro, options, and microstructure signals while preserving alignment and fallback behavior.

### Key components

- `features.pipeline`
  - Main feature assembly line used by training/prediction/backtest flows.
  - Exists to centralize feature definitions and ensure consistent columns between train and inference.
  - Includes base indicators, HAR/multi-scale signals, interaction terms, cross-asset and macro features, and optional intraday/LOB extensions.

- `features.options_factors`
  - Option surface factor extraction (ATM IV, term slope, skew, VRP-style proxies, etc.).
  - Exists because options markets often lead/confirm information not visible in price-only features.

- `features.research_factors`
  - Extended/experimental factor families for research depth.
  - Exists to isolate higher-complexity factor logic from the main pipeline and allow selective inclusion.

Other feature modules (`intraday`, `lob_features`, `macro`, `wave_flow`, `harx_spillovers`) exist to modularize domain-specific feature families and keep the main pipeline orchestration readable.

## Regime Detection Layer (`regime/`) — Detailed (Requested)

### Why regime detection exists in this system

The model is not assumed to behave uniformly across market states. Regime detection exists to:

- provide context features to the ML model,
- support regime-specific experts/blending,
- support risk suppression or de-leveraging in unstable conditions,
- quantify regime confidence (not just assign a hard label), and
- detect systemically different correlation environments (diversification breakdown risk).

### `regime.detector` (the orchestration layer)

Main role:
- Provides the system-facing regime API (`RegimeDetector`, `RegimeOutput`, `detect_regimes_batch`).

What it does:
- Supports two engines:
  - rule-based deterministic thresholds (legacy/fallback), and
  - HMM-based probabilistic regime inference.
- Produces:
  - `regime` labels,
  - `regime_confidence`,
  - posterior `regime_prob_*` columns,
  - `regime_duration`,
  - one-hot regime columns,
  - optional transition probability feature.

Why it is structured this way:
- The rule engine provides robustness/fallback when data is short or HMM fit fails.
- The HMM engine provides probabilistic state inference and smoother state transitions for real-world noisy data.

### Rule-based regime engine (inside `RegimeDetector._rule_detect`)

Intent:
- Deterministic, interpretable classification using technical state indicators when HMM is unavailable or unsuitable.

Signals used (with fallbacks):
- Hurst exponent (`Hurst_100`): trend vs mean-reversion tendency.
- ADX (`ADX_14`): trend strength.
- NATR (`NATR_14`): volatility proxy / volatility spike detection.
- `SMASlope_50`: directional trend sign.

Regime logic (canonical system regimes):
- `0` bull trend
- `1` bear trend
- `2` mean-reverting (default/neutral)
- `3` high volatility

Why confidence is still produced in rule mode:
- Downstream components expect confidence-aware behavior even when using deterministic labels.
- Confidence is derived from distance to decision boundaries / volatility z-scores.

### HMM regime engine (`RegimeDetector._hmm_detect` + `regime.hmm`)

Intent:
- Learn latent states from market observations and map them into system semantic regimes.

Why it exists:
- Markets transition between latent states that are not cleanly captured by fixed thresholds.
- Probabilistic state posteriors and sticky transitions are more realistic for noisy market data.

#### `regime.hmm.GaussianHMM`

What it is:
- In-repo Gaussian HMM implementation (no external HMM dependency required).

Why this custom implementation exists:
- Control over sticky transitions, duration smoothing, covariance mode, and numerical behavior.
- Fewer deployment dependencies.
- Regime semantics can be tailored to the rest of the system.

Important behaviors:
- EM/Baum-Welch fitting (`fit`)
- full or diagonal covariance support
- sticky transition shrinkage (discourages over-switching)
- Viterbi decoding for most likely state path
- duration smoothing (`_smooth_duration`) to merge very short runs
- BIC state-count selection (`select_hmm_states_bic`) when auto-select is enabled

#### Observation matrix construction (`build_hmm_observation_matrix`)

Why it exists:
- HMM needs a compact, numerically stable representation of market state.
- The function standardizes a small robust feature set (returns, vol, NATR, trend) to reduce instability and overfitting.

#### Raw-state-to-semantic mapping (`map_raw_states_to_regimes`)

Why it exists:
- HMM hidden state labels are arbitrary (state 0/1/2/3 are not semantically meaningful by default).
- The rest of the system expects canonical regime IDs with consistent meaning.

How it works conceptually:
- scores each raw HMM state by average return, volatility, trend, and Hurst-like tendency,
- assigns highest-volatility state to the high-vol regime first,
- then maps remaining states into bull/bear/mean-reversion semantics.

### Correlation regime detection (`regime.correlation`)

Intent:
- Detect a different failure mode than price-trend regimes: correlation spikes / diversification breakdown.

Why it exists:
- In stress, cross-asset correlations often rise sharply, making normal risk estimates and diversification assumptions unreliable.
- This signal improves the system's ability to adapt portfolio/risk logic beyond single-asset trend/vol states.

Key outputs:
- rolling average pairwise correlation (`avg_pairwise_corr`)
- binary correlation spike regime (`corr_regime`)
- correlation z-score (`corr_z_score`)

## Modeling Layer (`models/`)

### Core intent
- Train, version, load, and govern predictive models with regime-awareness and production-safe artifact handling.

### Key components

- `models.trainer`
  - Training orchestration and validation discipline.
  - Exists to keep the model development loop leakage-aware and reproducible.
  - Handles feature selection, purged CV, holdouts, ensemble/stacking, regime-specific experts, and artifact metadata.

- `models.predictor`
  - Runtime artifact loader + prediction engine.
  - Exists to standardize inference (version resolution, champion lookup, blending global/regime experts, confidence composition).

- `models.versioning`
  - Artifact registry/version utilities.
  - Exists so training and inference can coordinate on named versions/champions without ad-hoc path assumptions.

- `models.governance`
  - Policy layer around model promotion/replacement.
  - Exists to separate business rules from pure ML fitting code.

- `models.retrain_trigger`
  - Retrain decision logic.
  - Exists to trigger retraining for defined reasons (degradation/drift/schedule) rather than reactive manual runs.

- `models.iv.models` (UI-facing research subsystem)
  - IV analytics and surface modeling toolkit used by the Dash IV page.
  - Exists to support option pricing/Greeks, Heston/SVI calibration, arbitrage-aware SVI surface construction, and IV surface decomposition.

## Backtest + Execution Layer (`backtest/`)

### Core intent
- Convert predictive signals into a realistic simulated trading process with execution frictions, constraints, and risk overlays.

### Key components and why they exist

- `backtest.engine.Backtester`
  - Main simulation engine.
  - Exists to turn predictions into entries/exits while enforcing position limits, holding periods, execution costs, and risk controls.
  - Supports simple mode and full risk-managed mode.
  - Includes realistic execution enhancements (dynamic costs and optional Almgren-Chriss cost upgrade).
  - Tracks performance, turnover, regime breakdowns, and TCA-style outputs.

- `backtest.execution`
  - Execution cost/slippage utilities.
  - Exists to isolate execution assumptions from the strategy logic.

- `backtest.validation`
  - Standard validation metrics/reporting.
  - Exists to standardize backtest diagnostics.

- `backtest.advanced_validation`
  - Overfitting/robustness/capacity diagnostics (DSR, PBO, Monte Carlo, capacity analysis).
  - Exists to prevent promoting strategies on naive Sharpe alone.

## Risk Layer (`risk/`)

### Core intent
- Provide reusable, composable risk controls and analytics used in backtests, autopilot paper trading, and diagnostics.

### Main components

- `risk.position_sizer` — position sizing logic (including Kelly-style bounded sizing support).
- `risk.stop_loss` — stop policies and stop evaluation helpers.
- `risk.portfolio_risk` — portfolio-level risk checks/limits.
- `risk.drawdown` — drawdown tracking and control logic.
- `risk.metrics` — risk/performance metric calculations.
- `risk.covariance` — covariance estimation utilities.
- `risk.portfolio_optimizer` — portfolio weight optimization logic.
- `risk.stress_test` — scenario/stress testing analytics.
- `risk.attribution` — return/risk attribution helpers.
- `risk.factor_portfolio` — factor-based portfolio decomposition/management utilities.

Why this is separated from `backtest/`:
- It keeps risk logic reusable across simulation, autopilot, and analytics/UI contexts.

## Autopilot Layer (`autopilot/`)

### Core intent
- Run a disciplined strategy lifecycle on top of the predictive model instead of manually tweaking thresholds.

### Components

- `autopilot.strategy_discovery`
  - Deterministic candidate generation around execution-layer parameters (entry/confidence thresholds, risk mode, max positions).
  - Exists to search a bounded grid of strategy variants without changing the underlying predictor.

- `autopilot.promotion_gate`
  - Hard pass/fail gate + ranking score for deployability.
  - Exists to separate “candidate generation” from “promotion standards.”

- `autopilot.registry`
  - Persistent record of active promoted strategies and promotion history.
  - Exists for continuity across autopilot cycles and auditability.

- `autopilot.paper_trader`
  - Stateful paper execution engine.
  - Exists to test promoted strategies in a live-like loop before real deployment.
  - Includes cash accounting, entries/exits, bounded history, and optional Kelly-based sizing adjustments informed by recent paper trades and market risk stats.

- `autopilot.engine`
  - The orchestrator tying the above pieces together.
  - Exists so the autopilot cycle can be run as one controlled operational action.

## Kalshi Event-Market Vertical (`kalshi/`) — Detailed (Requested)

### Why this subsystem exists

This subsystem turns event-market quotes into machine-learning-ready event features and enforces the same validation/promotion rigor used in the core strategy stack.

### `kalshi.client` (API access, signing, routing, throttling)

Intent:
- Provide a production-safe HTTP client for Kalshi with signed authentication and endpoint routing.

Why each subcomponent exists:
- `KalshiSigner`
  - Generates RSA-PSS SHA256 request signatures in Kalshi's canonical message format.
  - Supports in-process cryptography signing with OpenSSL fallback to avoid runtime fragility.
- `RequestLimiter` + `RateLimitPolicy`
  - Prevents API abuse / throttling failures; can update limits from account-limit payloads.
- `KalshiDataRouter`
  - Routes requests to historical vs live endpoint roots based on cutoff timestamp and request parameters.
  - Exists because event research often spans both live and historical partitions.
- Retry policy / backoff
  - Handles 429/5xx transient failures robustly.

### `kalshi.storage` (event-time persistence schema)

Intent:
- Provide a stable event-time store (DuckDB preferred, SQLite fallback) for all Kalshi and macro-event research artifacts.

Why it exists:
- Event research needs reproducibility and audit trails.
- A consistent schema avoids “raw CSV sprawl” and makes feature generation deterministic.

What it stores (conceptually):
- market catalog + contract catalog + quote snapshots,
- macro event calendars and versioned snapshots,
- event outcomes (first print + revised),
- computed Kalshi distribution snapshots,
- event->market mapping versions,
- data provenance, coverage diagnostics, ingestion logs, daily health reports, and checkpoints.

### `kalshi.provider` (ingestion + retrieval + materialization orchestration)

Intent:
- Act as the high-level Kalshi data provider analogous to `WRDSProvider` for the equity stack.

Why it exists:
- To encapsulate the sequence of API calls, transformation, persistence, diagnostics materialization, and query helpers behind a single interface.

Key responsibilities:
- sync account limits / historical cutoff,
- sync market catalog and contracts,
- sync quote history,
- persist provenance + ingestion logs,
- compute/store distribution panels,
- materialize daily health reports,
- expose store-backed getters (markets/contracts/quotes/macro events/outcomes/mappings).

### `kalshi.distribution` (contract -> distribution reconstruction)

Intent:
- Convert per-contract quote states into a coherent market-level probability distribution snapshot with quality metadata.

Why it exists:
- Raw contract mids are not directly usable as event features.
- Event strategy features need moments, entropy, tails, and temporal change metrics.
- Real data requires repair logic (stale quotes, monotonicity violations, missing bins/tails, uncertain threshold direction).

What it does (conceptually):
- chooses latest quotes as-of a snapshot with dynamic stale cutoffs (time-to-event, market type, liquidity aware),
- resolves threshold direction semantics (`ge` vs `le`) with confidence levels,
- validates bins (overlaps/gaps/order),
- converts threshold or bin contracts into PMF-like mass,
- repairs monotonic threshold curves via isotonic projections when needed,
- computes moments (mean/var/skew), entropy, tail probabilities,
- computes quality dimensions and constraint-cleaning magnitudes,
- computes lagged distribution distance features (KL/JS/Wasserstein).

Why the extra metadata matters:
- The subsystem exposes cleaning magnitude, direction confidence, bin validity, and quality scores as first-class outputs so models can learn when the market representation is reliable versus noisy.

### `kalshi.events` (event-time feature/label builders)

Intent:
- Build leak-safe event-centric panels and labels around known macro event timestamps.

Why it exists:
- Event strategies need a different panel structure than continuous trading models: one row per (event, pre-release snapshot).

Core responsibilities:
- build pre-release snapshot grids from event calendars (`build_event_snapshot_grid`),
- strict backward as-of joining (`asof_join`),
- merge event->market mappings with event_id/event_type fallback,
- add revision-speed / variance-collapse / entropy-collapse features,
- build event labels from first print or revised outcomes with explicit label provenance,
- build asset-response labels in realistic post-event execution windows.

### `kalshi.walkforward` (event walk-forward + contract metrics)

Intent:
- Evaluate event feature panels under no-leakage, walk-forward splits and produce promotion-ready robustness metrics.

Why it exists:
- Event strategies have sparse samples and high multiple-testing risk; standard train/test splits are not enough.

What it produces:
- fold-by-fold IS/OOS correlation and return proxy metrics,
- purge/embargo-aware splits (including event-type-aware purge windows),
- advanced contract metrics (DSR, Monte Carlo robustness, bootstrap CI, capacity proxies, event-type stability),
- aggregated `EventWalkForwardResult` used directly in promotion.

### `kalshi.promotion` + `autopilot.promotion_gate`

Intent:
- Reuse the system-wide promotion discipline for event strategies while adding event-specific constraints.

Why it exists:
- Prevents the Kalshi vertical from becoming a separate lower-standard path to promotion.

## Dash UI (`dash_ui/`) — Detailed UI Component Audit (Requested)

### UI intent
- Provide a single operational/research interface for system monitoring, data inspection, model/regime diagnostics, backtest/risk analysis, IV surface modeling, S&P comparison, and autopilot/Kalshi workflows.

### UI architecture components (why each exists)

- `dash_ui/app.py`
  - App bootstrap and page registration orchestration.
  - Exists to compose the shell layout, page routing, and global stores/theme wiring.

- `dash_ui/server.py`
  - Runtime launcher / server wrapper.
  - Exists to separate app object construction from execution hosting concerns.

- `dash_ui/theme.py`
  - Centralized visual tokens and figure styling helpers.
  - Exists to keep page code focused on data/interaction instead of repeated style definitions.

- `dash_ui/assets/style.css`
  - Global CSS for page shell, responsive layout, and visual consistency.
  - Exists for styling concerns that are awkward/noisy in inline Dash styles.

- `dash_ui/components/*`
  - Reusable UI primitives and chart factories.
  - Exists to reduce page duplication and keep page modules focused on domain workflows.

### Shared UI component modules (current Dash UI)

- `dash_ui/components/alert_banner.py`: Alert banner component for displaying system messages and warnings. Classes: none. Functions: `alert_banner`.
- `dash_ui/components/chart_utils.py`: Plotly chart factory functions for the Quant Engine Dashboard. Classes: none. Functions: `line_chart`, `area_chart`, `bar_chart`, `heatmap_chart`, `surface_3d`, `equity_curve`, `regime_timeline`, `dual_axis_chart`, `candlestick_chart`, `scatter_chart`, `radar_chart`, `histogram_chart`.
- `dash_ui/components/health_check_list.py`: Reusable health check display component. Classes: none. Functions: `health_check_item`, `health_check_list`.
- `dash_ui/components/metric_card.py`: Reusable KPI metric card component. Classes: none. Functions: `metric_card`.
- `dash_ui/components/regime_badge.py`: Regime state badge component for displaying market regime indicators. Classes: none. Functions: `regime_badge`.
- `dash_ui/components/sidebar.py`: Sidebar navigation component with active-state highlighting. Classes: none. Functions: `_nav_item`, `create_sidebar`, `update_active_nav`.
- `dash_ui/components/status_bar.py`: Bottom status bar component. Classes: none. Functions: `create_status_bar`.
- `dash_ui/components/trade_table.py`: Styled DataTable component for displaying trades with conditional formatting. Classes: none. Functions: `trade_table`.

### Dash pages (all active UI pages)

- Total pages registered: 9
- Total callbacks detected: 31

### `Dashboard` (`/`)
- File: `dash_ui/pages/dashboard.py`
- Purpose (module doc): Dashboard -- Portfolio Intelligence Overview.
- UI IDs declared: 13
- Page helper/callback functions in file: 12
- Callback endpoints:
  - `load_dashboard_data` (line 175): outputs [dashboard-data.data]; inputs [dashboard-interval.n_intervals, dashboard-refresh-btn.n_clicks]
  - `update_metric_cards` (line 400): outputs [card-portfolio-value.children, card-30d-return.children, card-sharpe.children, card-regime.children, card-retrain.children, card-cv-gap.children, card-data-quality.children, card-system-health.children]; inputs [dashboard-data.data]
  - `render_tab_content` (line 456): outputs [dashboard-tab-content.children]; inputs [dashboard-tabs.value, dashboard-data.data]

### `System Health` (`/system-health`)
- File: `dash_ui/pages/system_health.py`
- Purpose (module doc): System Health Console -- comprehensive health assessment for the Quant Engine.
- UI IDs declared: 11
- Page helper/callback functions in file: 14
- Callback endpoints:
  - `load_health_data` (line 230): outputs [health-data.data]; inputs [health-interval.n_intervals, health-refresh-btn.n_clicks]
  - `update_health_cards` (line 285): outputs [hc-card-overall.children, hc-card-data.children, hc-card-promotion.children, hc-card-wf.children, hc-card-execution.children, hc-card-complexity.children]; inputs [health-data.data]
  - `render_health_tab` (line 344): outputs [health-tab-content.children]; inputs [health-tabs.value, health-data.data]

### `Data Explorer` (`/data-explorer`)
- File: `dash_ui/pages/data_explorer.py`
- Purpose (module doc): Data Explorer -- OHLCV visualization and quality analysis.
- UI IDs declared: 15
- Page helper/callback functions in file: 12
- Callback endpoints:
  - `load_data` (line 363): outputs [de-loaded-data.data, de-status-text.children]; inputs [de-load-btn.n_clicks]; state [de-universe-dropdown.value, de-timeframe-dropdown.value, de-ticker-input.value, de-loaded-data.data]
  - `render_ticker_list` (line 418): outputs [de-ticker-list-container.children]; inputs [de-loaded-data.data, de-selected-ticker.data]
  - `select_ticker` (line 471): outputs [de-selected-ticker.data]; inputs [n_clicks]; state [de-loaded-data.data]
  - `update_price_chart` (line 489): outputs [de-price-chart.figure]; inputs [de-selected-ticker.data]; state [de-loaded-data.data]
  - `update_volume_chart` (line 571): outputs [de-volume-chart.figure]; inputs [de-selected-ticker.data]; state [de-loaded-data.data]
  - `update_stats_bar` (line 613): outputs [de-stats-bar.children]; inputs [de-selected-ticker.data]; state [de-loaded-data.data]
  - `toggle_quality_modal` (line 666): outputs [de-quality-modal.is_open, de-quality-body.children]; inputs [de-quality-btn.n_clicks, de-quality-close-btn.n_clicks]; state [de-quality-modal.is_open, de-loaded-data.data]

### `Model Lab` (`/model-lab`)
- File: `dash_ui/pages/model_lab.py`
- Purpose (module doc): Model Lab -- Feature engineering, regime detection, and model training.
- UI IDs declared: 22
- Page helper/callback functions in file: 7
- Callback endpoints:
  - `load_features` (line 426): outputs [ml-feature-importance-chart.figure, ml-correlation-heatmap.figure, ml-features-status.children]; inputs [ml-load-features-btn.n_clicks]
  - `run_regime_detection` (line 576): outputs [ml-regime-timeline.figure, ml-regime-probs.figure, ml-transition-matrix.figure, ml-regime-status.children]; inputs [ml-run-regime-btn.n_clicks]
  - `train_model` (line 741): outputs [ml-cv-results-chart.figure, ml-model-summary.children, ml-train-progress.value, ml-train-status.children]; inputs [ml-train-btn.n_clicks]; state [ml-train-universe.value, ml-train-horizon.value, ml-train-feature-mode.value, ml-train-options.value]

### `Signal Desk` (`/signal-desk`)
- File: `dash_ui/pages/signal_desk.py`
- Purpose (module doc): Signal Desk -- Prediction generation and signal ranking.
- UI IDs declared: 9
- Page helper/callback functions in file: 3
- Callback endpoints:
  - `generate_signals` (line 306): outputs [sd-signal-table-container.children, sd-distribution-chart.figure, sd-scatter-chart.figure, sd-status-text.children, sd-signals-data.data]; inputs [sd-generate-btn.n_clicks]; state [sd-horizon-dropdown.value, sd-topn-input.value]

### `Backtest & Risk` (`/backtest-risk`)
- File: `dash_ui/pages/backtest_risk.py`
- Purpose (module doc): Backtest & Risk -- Equity curves, risk metrics, and trade analysis.
- UI IDs declared: 16
- Page helper/callback functions in file: 5
- Callback endpoints:
  - `run_backtest` (line 353): outputs [bt-trades-store.data, bt-returns-store.data, bt-total-return.children, bt-ann-return.children, bt-sharpe.children, bt-sortino.children, bt-max-dd.children, bt-win-rate.children, bt-profit-factor.children, bt-total-trades.children, bt-equity-curve.figure, bt-drawdown-chart.figure]; inputs [bt-run-btn.n_clicks]; state [bt-holding-period.value, bt-max-positions.value, bt-entry-threshold.value, bt-position-size.value, bt-risk-mgmt.value]
  - `update_risk_analytics` (line 554): outputs [risk-var-waterfall.figure, risk-metrics-text.children, risk-rolling-chart.figure, risk-return-dist.figure]; inputs [bt-returns-store.data]
  - `update_trade_analysis` (line 712): outputs [bt-trade-table.data, bt-regime-perf.figure]; inputs [bt-trades-store.data]

### `IV Surface` (`/iv-surface`)
- File: `dash_ui/pages/iv_surface.py`
- Purpose (module doc): IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling.
- UI IDs declared: 15
- Page helper/callback functions in file: 10
- Callback endpoints:
  - `set_svi_preset` (line 369): outputs [svi-a.value, svi-b.value, svi-rho.value, svi-m.value, svi-sigma.value]; inputs [svi-preset-dd.value]
  - `update_svi_surface` (line 388): outputs [svi-surface-3d.figure, svi-smiles-2d.figure]; inputs [svi-a.value, svi-b.value, svi-rho.value, svi-m.value, svi-sigma.value]
  - `set_heston_preset` (line 422): outputs [heston-v0.value, heston-theta.value, heston-kappa.value, heston-sigma.value, heston-rho.value]; inputs [heston-preset-dd.value]
  - `compute_heston_surface` (line 443): outputs [heston-surface-3d.figure, heston-smiles-2d.figure]; inputs [heston-compute-btn.n_clicks]; state [heston-v0.value, heston-theta.value, heston-kappa.value, heston-sigma.value, heston-rho.value]
  - `build_arb_free_surface` (line 562): outputs [arb-surface-3d.figure, arb-smiles-2d.figure]; inputs [arb-build-btn.n_clicks]; state [arb-spot.value, arb-rate.value, arb-div.value, arb-noise.value, arb-max-iter.value]

### `S&P Comparison` (`/sp-comparison`)
- File: `dash_ui/pages/sp_comparison.py`
- Purpose (module doc): S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation.
- UI IDs declared: 11
- Page helper/callback functions in file: 3
- Callback endpoints:
  - `load_and_compare` (line 161): outputs [sp-data-store.data, sp-strat-return.children, sp-bench-return.children, sp-alpha.children, sp-beta.children, sp-correlation.children, sp-tracking-error.children, sp-info-ratio.children, sp-equity-chart.figure, sp-corr-chart.figure, sp-alpha-beta-chart.figure, sp-relative-chart.figure, sp-dd-chart.figure, sp-frame-store.data]; inputs [sp-load-btn.n_clicks]; state [sp-period-dd.value]
  - `toggle_animation` (line 433): outputs [animation-interval.disabled, sp-animate-btn.children]; inputs [sp-animate-btn.n_clicks]; state [animation-interval.disabled]
  - `animate_equity_chart` (line 452): outputs [sp-equity-chart.figure, sp-frame-store.data]; inputs [animation-interval.n_intervals]; state [sp-data-store.data, sp-frame-store.data]

### `Autopilot & Events` (`/autopilot`)
- File: `dash_ui/pages/autopilot_kalshi.py`
- Purpose (module doc): Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets.
- UI IDs declared: 14
- Page helper/callback functions in file: 8
- Callback endpoints:
  - `update_autopilot` (line 424): outputs [ap-strategy-table.data, ap-funnel-chart.figure, ap-feature-mode-badge.children]; inputs [ap-discover-btn.n_clicks, ap-registry-btn.n_clicks]
  - `load_paper_trading` (line 520): outputs [paper-equity-chart.figure, paper-positions-table.data]; inputs [paper-load-btn.n_clicks]
  - `load_kalshi_events` (line 596): outputs [kalshi-prob-chart.figure, kalshi-timeline-chart.figure, kalshi-wf-chart.figure, kalshi-disagree-chart.figure]; inputs [kalshi-load-btn.n_clicks]; state [kalshi-event-dd.value]


### Why the UI is page-heavy instead of a single monolith

- Each page maps to a distinct operational question:
  - “Is the system healthy?”
  - “What data do I actually have?”
  - “What regime/model state am I in?”
  - “What would a strategy do under these parameters?”
  - “What is autopilot doing?”
- This reduces coupling and makes it easier for an LLM (or a human) to modify one workflow without breaking another.

## Tests as System Specification (`tests/` and `kalshi/tests/`)

These tests are not just correctness checks; they encode design intent and system contracts.

Examples of what the tests reveal about system intent:
- leakage prevention is mandatory (`test_panel_split`, `kalshi` no-leakage/purge tests),
- delisting return correctness matters (`tests/test_delisting_total_return.py`),
- execution cost realism matters (`tests/test_execution_dynamic_costs.py`),
- autopilot fallback behavior is expected and supported (`tests/test_autopilot_predictor_fallback.py`),
- Kalshi distribution hardening is a formalized requirement (`kalshi/tests/test_threshold_direction.py`, `test_bin_validity.py`, `test_stale_quotes.py`, `test_signature_kat.py`).

## Cleanup and Repo Organization Changes Applied in This Audit

### Legacy UI removal (safe migration to Dash UI)

Removed:
- `run_dashboard.py`
- legacy `ui/` package (all page/component/theme modules under that tree)

Safety checks performed:
- Python import/reference scan after deletion shows no runtime imports of `ui` remain in active code.
- Remaining mention is a harmless historical comment in `dash_ui/theme.py` noting palette provenance.

Legacy UI status: **removed**

### Root file organization (moved into `docs/`)

Created folders:
- `docs/guides`
- `docs/reports`
- `docs/plans`
- `docs/notes`

Moved files:
- `DASH_FOUNDATION_SUMMARY.md` -> `docs/guides/DASH_FOUNDATION_SUMMARY.md`
- `DASH_QUICK_START.md` -> `docs/guides/DASH_QUICK_START.md`
- `UI_IMPROVEMENT_GUIDE.md` -> `docs/guides/UI_IMPROVEMENT_GUIDE.md`
- `DATA_INTEGRATION_REPORT.md` -> `docs/reports/DATA_INTEGRATION_REPORT.md`
- `FINAL_AUDIT_FINDINGS.md` -> `docs/reports/FINAL_AUDIT_FINDINGS.md`
- `SYSTEM_AUDIT_REPORT.md` -> `docs/reports/SYSTEM_AUDIT_REPORT.md`
- `WRDS_DAILY_REDOWNLOAD_REPORT.md` -> `docs/reports/WRDS_DAILY_REDOWNLOAD_REPORT.md`
- `IMPROVEMENT_ROADMAP.md` -> `docs/plans/IMPROVEMENT_ROADMAP.md`
- `instructions` -> `docs/notes/instructions`

### Duplicate-file check (exact content)

- Exact duplicate content group: `backtest/__init__.py`, `features/__init__.py`

Interpretation:
- These are benign duplicate `__init__.py` stubs (common in Python packages).
- No exact duplicate reports, source modules, or data artifacts were found in the current repo tree.

## Current Top-Level Repo Layout (Post-Cleanup)

- `.claude`
- `__init__.py`
- `autopilot`
- `backtest`
- `config.py`
- `dash_ui`
- `data`
- `docs`
- `features`
- `indicators`
- `kalshi`
- `models`
- `regime`
- `reproducibility.py`
- `results`
- `risk`
- `run_autopilot.py`
- `run_backtest.py`
- `run_dash.py`
- `run_kalshi_event_pipeline.py`
- `run_predict.py`
- `run_rehydrate_cache_metadata.py`
- `run_retrain.py`
- `run_train.py`
- `run_wrds_daily_refresh.py`
- `tests`
- `trained_models`
- `utils`

## How To Use This Document With Another LLM

Recommended prompt pattern:
- “Read `/Users/justinblaise/Documents/FRAMEWORK SCRIPTS/quant_engine/docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` first. Treat it as the architecture and intent contract. Then inspect only the modules relevant to [task]. Preserve PERMNO-first identity, no-leakage joins, promotion gates, and Dash UI page separation.”

This helps the next model avoid:
- reintroducing ticker-first joins,
- leakage in event features/walk-forward splits,
- bypassing promotion quality gates,
- touching the wrong UI stack (legacy `ui` is removed; `dash_ui` is active).

---

# Appendix A: Exhaustive Component Inventory (Current Source Tree)

This appendix is a source-derived inventory of modules, classes, and top-level functions, with docstring-based intent summaries where available.

## Package `(root)`
- Modules: 12
### `__init__.py` (6 lines)
- Intent: Quant Engine - Continuous Feature ML Trading System
- Classes: none
- Top-level functions: none

### `config.py` (271 lines)
- Intent: Central configuration for the quant engine.
- Classes: none
- Top-level functions: none

### `reproducibility.py` (333 lines)
- Intent: Reproducibility locks for run manifests.
- Classes: none
- Top-level functions:
  - `_get_git_commit`: Return the current git commit hash, or 'unknown' if not in a repo.
  - `_dataframe_checksum`: Compute a lightweight checksum of a DataFrame's shape and sample.
  - `build_run_manifest`: Build a reproducibility manifest for a pipeline run.
  - `write_run_manifest`: Write manifest to JSON file. Returns the output path.
  - `verify_manifest`: Verify current environment matches a stored manifest.
  - `replay_manifest`: Re-run a historical cycle and compare to stored results.

### `run_autopilot.py` (88 lines)
- Intent: Run one full autopilot cycle:
- Classes: none
- Top-level functions:
  - `main`: No function docstring.

### `run_backtest.py` (426 lines)
- Intent: Backtest the trained model on historical data.
- Classes: none
- Top-level functions:
  - `main`: No function docstring.

### `run_dash.py` (82 lines)
- Intent: Quant Engine -- Dash Dashboard Launcher.
- Classes: none
- Top-level functions:
  - `_check_dependencies`: Return list of missing package install names.
  - `main`: Launch the Quant Engine Dash Dashboard.

### `run_kalshi_event_pipeline.py` (225 lines)
- Intent: Run the integrated Kalshi event-time pipeline inside quant_engine.
- Classes: none
- Top-level functions:
  - `_read_df`: No function docstring.
  - `main`: No function docstring.

### `run_predict.py` (201 lines)
- Intent: Generate predictions using trained ensemble model.
- Classes: none
- Top-level functions:
  - `main`: No function docstring.

### `run_rehydrate_cache_metadata.py` (99 lines)
- Intent: Backfill cache metadata sidecars for existing OHLCV cache files.
- Classes: none
- Top-level functions:
  - `_parse_root_source`: No function docstring.
  - `main`: No function docstring.

### `run_retrain.py` (289 lines)
- Intent: Retrain the quant engine model — checks triggers and retrains if needed.
- Classes: none
- Top-level functions:
  - `_check_regime_change_trigger`: Check whether the market regime has changed for a sustained period.
  - `main`: No function docstring.

### `run_train.py` (194 lines)
- Intent: Train the regime-conditional ensemble model.
- Classes: none
- Top-level functions:
  - `main`: No function docstring.

### `run_wrds_daily_refresh.py` (347 lines)
- Intent: Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
- Classes: none
- Top-level functions:
  - `_build_ticker_list`: Build the full ticker list from cached + UNIVERSE_FULL + BENCHMARK.
  - `_verify_file`: Verify OHLCV quality for a single parquet file. Returns dict of results.
  - `_verify_all`: Run verification on all _1d.parquet files in cache.
  - `_cleanup_old_daily`: Remove old {TICKER}_daily_{dates}.parquet and .meta.json files.
  - `main`: No function docstring.

## Package `autopilot`
- Modules: 6
### `autopilot/__init__.py` (20 lines)
- Intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- Classes: none
- Top-level functions: none

### `autopilot/engine.py` (910 lines)
- Intent: End-to-end autopilot cycle:
- Classes:
  - `HeuristicPredictor`: Lightweight fallback predictor used when sklearn-backed model artifacts Methods: `__init__`, `_rolling_zscore`, `predict`.
  - `AutopilotEngine`: Coordinates discovery, promotion, and paper execution. Methods: `__init__`, `_log`, `_is_permno_key`, `_assert_permno_price_data`, `_assert_permno_prediction_panel`, `_assert_permno_latest_predictions`, `_load_data`, `_build_regimes`, `_train_baseline`, `_ensure_predictor`, `_predict_universe`, `_walk_forward_predictions`, `_evaluate_candidates`, `_compute_optimizer_weights`, `run_cycle`.
- Top-level functions: none

### `autopilot/paper_trader.py` (432 lines)
- Intent: Stateful paper-trading engine for promoted strategies.
- Classes:
  - `PaperTrader`: Executes paper entries/exits from promoted strategy definitions. Methods: `__init__`, `_load_state`, `_save_state`, `_resolve_as_of`, `_latest_predictions_by_id`, `_latest_predictions_by_ticker`, `_current_price`, `_position_id`, `_mark_to_market`, `_trade_return`, `_historical_trade_stats`, `_market_risk_stats`, `_position_size_pct`, `run_cycle`.
- Top-level functions: none

### `autopilot/promotion_gate.py` (230 lines)
- Intent: Promotion gate for deciding whether a discovered strategy is deployable.
- Classes:
  - `PromotionDecision`: No class docstring. Methods: `to_dict`.
  - `PromotionGate`: Applies hard risk/quality constraints before a strategy can be paper-deployed. Methods: `__init__`, `evaluate`, `evaluate_event_strategy`, `rank`.
- Top-level functions: none

### `autopilot/registry.py` (103 lines)
- Intent: Persistent strategy registry for promoted candidates.
- Classes:
  - `ActiveStrategy`: No class docstring. Methods: `to_dict`.
  - `StrategyRegistry`: Maintains promoted strategy state and historical promotion decisions. Methods: `__init__`, `_load`, `_save`, `get_active`, `apply_promotions`.
- Top-level functions: none

### `autopilot/strategy_discovery.py` (75 lines)
- Intent: Strategy discovery for execution-layer parameter variants.
- Classes:
  - `StrategyCandidate`: No class docstring. Methods: `to_dict`.
  - `StrategyDiscovery`: Generates a deterministic candidate grid for backtest validation. Methods: `__init__`, `generate`.
- Top-level functions: none

## Package `backtest`
- Modules: 6
### `backtest/__init__.py` (0 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes: none
- Top-level functions: none

### `backtest/advanced_validation.py` (551 lines)
- Intent: Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.
- Classes:
  - `DeflatedSharpeResult`: Result of Deflated Sharpe Ratio test. Methods: none.
  - `PBOResult`: Probability of Backtest Overfitting result. Methods: none.
  - `MonteCarloResult`: Monte Carlo simulation result. Methods: none.
  - `CapacityResult`: Strategy capacity analysis. Methods: none.
  - `AdvancedValidationReport`: Complete advanced validation report. Methods: none.
- Top-level functions:
  - `deflated_sharpe_ratio`: Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
  - `probability_of_backtest_overfitting`: Probability of Backtest Overfitting (Bailey et al., 2017).
  - `monte_carlo_validation`: Monte Carlo validation of strategy performance.
  - `capacity_analysis`: Estimate strategy capacity and market impact.
  - `run_advanced_validation`: Run all advanced validation tests.
  - `_print_report`: Pretty-print advanced validation report.

### `backtest/engine.py` (1625 lines)
- Intent: Backtester — converts model predictions into simulated trades.
- Classes:
  - `Trade`: No class docstring. Methods: none.
  - `BacktestResult`: No class docstring. Methods: none.
  - `Backtester`: Simulates trading from model predictions. Methods: `__init__`, `_init_risk_components`, `_almgren_chriss_cost_bps`, `_simulate_entry`, `_simulate_exit`, `_execution_context`, `_effective_return_series`, `_delisting_adjustment_multiplier`, `_trade_realized_return`, `_is_permno_key`, `_assert_permno_inputs`, `run`, `_process_signals`, `_process_signals_risk_managed`, `_compute_metrics`, `_build_daily_equity`, `_compute_turnover`, `_compute_regime_performance`, `_compute_tca`, `_print_result`, `_empty_result`.
- Top-level functions: none

### `backtest/execution.py` (271 lines)
- Intent: Execution simulator with spread, market impact, and participation limits.
- Classes:
  - `ExecutionFill`: No class docstring. Methods: none.
  - `ExecutionModel`: Simple market-impact model for backtests. Methods: `__init__`, `simulate`.
- Top-level functions:
  - `calibrate_cost_model`: Calibrate execution-cost model parameters from historical fills.

### `backtest/optimal_execution.py` (199 lines)
- Intent: Almgren-Chriss (2001) optimal execution model.
- Classes: none
- Top-level functions:
  - `almgren_chriss_trajectory`: Compute the optimal execution trajectory using the Almgren-Chriss model.
  - `estimate_execution_cost`: Estimate the total execution cost for a given trade trajectory.

### `backtest/validation.py` (726 lines)
- Intent: Walk-forward validation and statistical tests.
- Classes:
  - `WalkForwardFold`: No class docstring. Methods: none.
  - `WalkForwardResult`: No class docstring. Methods: none.
  - `StatisticalTests`: No class docstring. Methods: none.
  - `CPCVResult`: No class docstring. Methods: none.
  - `SPAResult`: No class docstring. Methods: none.
- Top-level functions:
  - `walk_forward_validate`: Walk-forward validation of prediction quality with purge gap.
  - `_benjamini_hochberg`: Benjamini-Hochberg procedure for multiple testing correction.
  - `run_statistical_tests`: Statistical tests for prediction quality.
  - `_partition_bounds`: Return contiguous [start, end) bounds for temporal partitions.
  - `combinatorial_purged_cv`: Combinatorial Purged Cross-Validation for signal robustness.
  - `strategy_signal_returns`: Build per-sample strategy return series from prediction signals.
  - `superior_predictive_ability`: Single-strategy SPA-style block-bootstrap test on differential returns.

## Package `dash_ui`
- Modules: 26
### `dash_ui/__init__.py` (23 lines)
- Intent: Quantitative Trading Engine - Dash UI Package
- Classes: none
- Top-level functions: none

### `dash_ui/app.py` (160 lines)
- Intent: Quant Engine Professional Dashboard — Dash Application Factory.
- Classes: none
- Top-level functions:
  - `create_app`: Create and configure the Dash application with multi-page support.

### `dash_ui/components/__init__.py` (34 lines)
- Intent: Reusable Dash components for the Quant Engine Dashboard.
- Classes: none
- Top-level functions: none

### `dash_ui/components/alert_banner.py` (130 lines)
- Intent: Alert banner component for displaying system messages and warnings.
- Classes: none
- Top-level functions:
  - `alert_banner`: Create a styled alert banner message.

### `dash_ui/components/chart_utils.py` (645 lines)
- Intent: Plotly chart factory functions for the Quant Engine Dashboard.
- Classes: none
- Top-level functions:
  - `line_chart`: Create a multi-line chart.
  - `area_chart`: Create a filled area chart.
  - `bar_chart`: Create a bar chart (vertical or horizontal).
  - `heatmap_chart`: Create an annotated heatmap (e.g., correlation matrix).
  - `surface_3d`: Create a 3D surface plot.
  - `equity_curve`: Create an equity curve chart with drawdown shading.
  - `regime_timeline`: Create a regime timeline with colored vertical bands.
  - `dual_axis_chart`: Create a chart with two y-axes (left and right).
  - `candlestick_chart`: Create a candlestick chart with volume subplot.
  - `scatter_chart`: Create a scatter plot with optional color/size encoding.
  - `radar_chart`: Create a radar/polar chart.
  - `histogram_chart`: Create a distribution histogram.

### `dash_ui/components/health_check_list.py` (38 lines)
- Intent: Reusable health check display component.
- Classes: none
- Top-level functions:
  - `health_check_item`: Render a single health check row with icon, label, and optional detail.
  - `health_check_list`: Render a list of health checks.

### `dash_ui/components/metric_card.py` (46 lines)
- Intent: Reusable KPI metric card component.
- Classes: none
- Top-level functions:
  - `metric_card`: Create a styled KPI metric card.

### `dash_ui/components/regime_badge.py` (61 lines)
- Intent: Regime state badge component for displaying market regime indicators.
- Classes: none
- Top-level functions:
  - `regime_badge`: Create a colored badge displaying the regime name.

### `dash_ui/components/sidebar.py` (121 lines)
- Intent: Sidebar navigation component with active-state highlighting.
- Classes: none
- Top-level functions:
  - `_nav_item`: Render a single navigation item.
  - `create_sidebar`: Create the full sidebar layout.
  - `update_active_nav`: Highlight the active navigation item based on URL.

### `dash_ui/components/status_bar.py` (34 lines)
- Intent: Bottom status bar component.
- Classes: none
- Top-level functions:
  - `create_status_bar`: Create the bottom status bar.

### `dash_ui/components/trade_table.py` (226 lines)
- Intent: Styled DataTable component for displaying trades with conditional formatting.
- Classes: none
- Top-level functions:
  - `trade_table`: Create a styled DataTable for displaying trade-level data.

### `dash_ui/data/__init__.py` (1 lines)
- Intent: Data loading and caching layer for the Dash UI.
- Classes: none
- Top-level functions: none

### `dash_ui/data/cache.py` (91 lines)
- Intent: Caching layer for Dash UI data loading operations.
- Classes: none
- Top-level functions:
  - `init_cache`: Initialize the Flask cache instance with a Dash application.
  - `cached`: Decorator to cache function results for a specified duration.

### `dash_ui/data/loaders.py` (726 lines)
- Intent: Data loading and computation functions for the Dash UI.
- Classes:
  - `HealthCheck`: Single health check result. Methods: none.
  - `SystemHealthPayload`: Full system health assessment. Methods: none.
- Top-level functions:
  - `load_trades`: Load and clean backtest trade CSV.
  - `build_portfolio_returns`: Build daily portfolio returns from trade-level data.
  - `_read_close_returns`: Read close returns from a parquet file.
  - `load_benchmark_returns`: Load benchmark (SPY) returns from parquet cache.
  - `load_factor_proxies`: Load factor proxy returns for attribution.
  - `compute_risk_metrics`: Compute portfolio risk metrics from daily returns.
  - `compute_regime_payload`: Run HMM regime detection and return structured results.
  - `compute_model_health`: Assess model health from registry and trade data.
  - `load_feature_importance`: Load feature importance from latest model metadata.
  - `compute_health_scores`: Quick health indicators for dashboard cards.
  - `score_to_status`: Convert numeric score to PASS/WARN/FAIL status.
  - `collect_health_data`: Run full system health assessment.
  - `_check_data_integrity`: Check survivorship bias and data quality.
  - `_check_promotion_contract`: Verify promotion gate configuration.
  - `_check_walkforward`: Verify walk-forward validation setup.
  - `_check_execution`: Audit execution cost model.
  - `_check_complexity`: Audit feature and knob complexity.
  - `_check_strengths`: Identify what's working well.

### `dash_ui/pages/__init__.py` (26 lines)
- Intent: Dash page modules for the Quant Engine Dashboard.
- Classes: none
- Top-level functions: none

### `dash_ui/pages/autopilot_kalshi.py` (771 lines)
- Intent: Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets.
- Classes: none
- Top-level functions:
  - `_demo_strategy_candidates`: Generate realistic demo strategy candidate data.
  - `_demo_promotion_funnel`: Demo promotion funnel counts.
  - `_demo_paper_equity`: Generate demo paper trading equity curve.
  - `_demo_paper_positions`: Generate demo paper trading positions.
  - `_demo_kalshi_events`: Generate demo Kalshi event data for a given type.
  - `update_autopilot`: Run discovery cycle or load registry.
  - `load_paper_trading`: Load paper trading state or use demo data.
  - `load_kalshi_events`: Load Kalshi event data or use demo data.

### `dash_ui/pages/backtest_risk.py` (787 lines)
- Intent: Backtest & Risk -- Equity curves, risk metrics, and trade analysis.
- Classes: none
- Top-level functions:
  - `_control_group`: No function docstring.
  - `_metrics_row`: No function docstring.
  - `run_backtest`: Load existing backtest results or run a new backtest.
  - `update_risk_analytics`: Compute and display risk analytics from stored returns.
  - `update_trade_analysis`: Populate trade table and regime performance chart.

### `dash_ui/pages/dashboard.py` (1040 lines)
- Intent: Dashboard -- Portfolio Intelligence Overview.
- Classes: none
- Top-level functions:
  - `_card_panel`: Wrap children inside a styled card-panel div.
  - `_pct`: No function docstring.
  - `_fmt`: No function docstring.
  - `load_dashboard_data`: Load all data and cache in dcc.Store as JSON-safe dict.
  - `update_metric_cards`: No function docstring.
  - `render_tab_content`: No function docstring.
  - `_render_portfolio_tab`: No function docstring.
  - `_render_regime_tab`: No function docstring.
  - `_render_model_tab`: No function docstring.
  - `_render_features_tab`: No function docstring.
  - `_render_trades_tab`: No function docstring.
  - `_render_risk_tab`: No function docstring.

### `dash_ui/pages/data_explorer.py` (795 lines)
- Intent: Data Explorer -- OHLCV visualization and quality analysis.
- Classes: none
- Top-level functions:
  - `_download_ticker`: Load OHLCV data for a ticker and timeframe.
  - `_generate_demo_data`: Generate realistic synthetic OHLCV data seeded by ticker name.
  - `_compute_sma`: Simple moving average with NaN fill for insufficient data.
  - `_df_to_store`: Serialize a DataFrame to a JSON-safe dict for dcc.Store.
  - `_store_to_df`: Deserialize a dict from dcc.Store back to a DataFrame.
  - `load_data`: No function docstring.
  - `render_ticker_list`: No function docstring.
  - `select_ticker`: No function docstring.
  - `update_price_chart`: No function docstring.
  - `update_volume_chart`: No function docstring.
  - `update_stats_bar`: No function docstring.
  - `toggle_quality_modal`: No function docstring.

### `dash_ui/pages/iv_surface.py` (677 lines)
- Intent: IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling.
- Classes: none
- Top-level functions:
  - `compute_svi_surface`: Compute SVI implied volatility surface analytically.
  - `compute_svi_smiles`: Compute individual smile curves for select expiries.
  - `_svi_slider`: No function docstring.
  - `_build_surface_figure`: Create a go.Surface figure with Inferno colorscale.
  - `_build_smiles_figure`: Create 2D smile curves for select expiries.
  - `set_svi_preset`: Set slider values from preset selection.
  - `update_svi_surface`: Recompute SVI surface from slider values (analytical, fast).
  - `set_heston_preset`: Set Heston slider values from preset.
  - `compute_heston_surface`: Compute Heston IV surface (computationally intensive).
  - `build_arb_free_surface`: Build arbitrage-free SVI surface from synthetic market data.

### `dash_ui/pages/model_lab.py` (919 lines)
- Intent: Model Lab -- Feature engineering, regime detection, and model training.
- Classes: none
- Top-level functions:
  - `_demo_feature_importance`: Generate plausible demo feature importance data.
  - `load_features`: No function docstring.
  - `_demo_regime_payload`: Generate plausible demo regime data.
  - `run_regime_detection`: No function docstring.
  - `train_model`: No function docstring.
  - `_build_cv_chart`: Build a grouped bar chart comparing CV folds and holdout.
  - `_build_demo_cv_chart`: Build a demo CV chart with synthetic data.

### `dash_ui/pages/signal_desk.py` (536 lines)
- Intent: Signal Desk -- Prediction generation and signal ranking.
- Classes: none
- Top-level functions:
  - `_generate_demo_signals`: Generate realistic demo signal data for the given tickers.
  - `_try_live_signals`: Attempt to generate real signals using the model pipeline.
  - `generate_signals`: No function docstring.

### `dash_ui/pages/sp_comparison.py` (501 lines)
- Intent: S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation.
- Classes: none
- Top-level functions:
  - `load_and_compare`: Load portfolio and benchmark returns, compute comparison metrics.
  - `toggle_animation`: Toggle the animation interval on/off.
  - `animate_equity_chart`: Progressively reveal data points on the main equity chart.

### `dash_ui/pages/system_health.py` (1081 lines)
- Intent: System Health Console -- comprehensive health assessment for the Quant Engine.
- Classes: none
- Top-level functions:
  - `_card_panel`: Wrap children inside a styled card-panel div.
  - `_status_span`: Return a colored status span with icon.
  - `_check_row`: Render a single health check as a row.
  - `_instruction_banner`: Amber instruction banner at top of a tab.
  - `_score_color`: Return color based on score threshold.
  - `load_health_data`: Run collect_health_data() and serialize to JSON-safe dict.
  - `update_health_cards`: No function docstring.
  - `render_health_tab`: No function docstring.
  - `_render_overview_tab`: No function docstring.
  - `_render_data_tab`: No function docstring.
  - `_render_promotion_tab`: No function docstring.
  - `_render_wf_tab`: No function docstring.
  - `_render_execution_tab`: No function docstring.
  - `_render_complexity_tab`: No function docstring.

### `dash_ui/server.py` (52 lines)
- Intent: Entry point for the Quant Engine Dash application.
- Classes: none
- Top-level functions: none

### `dash_ui/theme.py` (205 lines)
- Intent: Bloomberg-inspired dark theme for the Quant Engine Dash Dashboard.
- Classes: none
- Top-level functions:
  - `apply_plotly_template`: Register and activate the Bloomberg dark Plotly template.
  - `create_figure`: Create a Plotly figure with the Bloomberg dark template applied.
  - `empty_figure`: Return an empty Plotly figure with a centered message.
  - `format_pct`: Format a decimal value as percentage string.
  - `format_number`: Format a number with thousand separators and decimal places.
  - `metric_color`: Return color hex code based on value sign and preference.

## Package `data`
- Modules: 10
### `data/__init__.py` (13 lines)
- Intent: Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
- Classes: none
- Top-level functions: none

### `data/alternative.py` (652 lines)
- Intent: Alternative data framework — WRDS-backed implementation.
- Classes:
  - `AlternativeDataProvider`: WRDS-backed alternative data provider. Methods: `__init__`, `_resolve_permno`, `get_earnings_surprise`, `get_options_flow`, `get_short_interest`, `get_insider_transactions`, `get_institutional_ownership`.
- Top-level functions:
  - `_get_wrds`: Return the cached WRDSProvider singleton, or None.
  - `compute_alternative_features`: Gather all available alternative data and return as a feature DataFrame.

### `data/feature_store.py` (308 lines)
- Intent: Point-in-time feature store for backtest acceleration.
- Classes:
  - `FeatureStore`: Point-in-time feature store for backtest acceleration. Methods: `__init__`, `_version_dir`, `_ts_tag`, `_parquet_path`, `_meta_path`, `save_features`, `load_features`, `list_available`, `invalidate`.
- Top-level functions: none

### `data/loader.py` (707 lines)
- Intent: Data loader — self-contained data loading with multiple sources.
- Classes: none
- Top-level functions:
  - `_permno_from_meta`: No function docstring.
  - `_ticker_from_meta`: No function docstring.
  - `_attach_id_attrs`: No function docstring.
  - `_cache_source`: No function docstring.
  - `_cache_is_usable`: No function docstring.
  - `_cached_universe_subset`: Prefer locally cached symbols to keep offline runs deterministic.
  - `_normalize_ohlcv`: Return a sorted, deterministic OHLCV frame or None if invalid.
  - `_harmonize_return_columns`: Standardize return columns so backtests can consume total-return streams.
  - `_merge_option_surface_from_prefetch`: Merge pre-fetched OptionMetrics surface rows into a single PERMNO panel.
  - `load_ohlcv`: Load daily OHLCV data for a single ticker.
  - `load_universe`: Load OHLCV data for multiple symbols. Returns {permno: DataFrame}.
  - `load_survivorship_universe`: Load a survivorship-bias-free universe using WRDS CRSP.
  - `load_with_delistings`: Load OHLCV data including delisting returns from CRSP.

### `data/local_cache.py` (674 lines)
- Intent: Local data cache for daily OHLCV data.
- Classes: none
- Top-level functions:
  - `_ensure_cache_dir`: Create cache directory if it doesn't exist.
  - `_normalize_ohlcv_columns`: Normalize OHLCV column names to quant_engine's canonical schema.
  - `_to_daily_ohlcv`: Convert any candidate frame into validated daily OHLCV.
  - `_read_csv_ohlcv`: No function docstring.
  - `_candidate_csv_paths`: No function docstring.
  - `_cache_meta_path`: No function docstring.
  - `_read_cache_meta`: No function docstring.
  - `_write_cache_meta`: No function docstring.
  - `save_ohlcv`: Save OHLCV DataFrame to local cache.
  - `load_ohlcv_with_meta`: Load OHLCV and sidecar metadata from cache roots.
  - `load_ohlcv`: Load OHLCV DataFrame from local cache.
  - `load_intraday_ohlcv`: Load intraday OHLCV data from cache.
  - `list_intraday_timeframes`: Return list of available intraday timeframes for a ticker in the cache.
  - `list_cached_tickers`: List all tickers available in cache roots.
  - `_daily_cache_files`: Return de-duplicated daily-cache candidate files for one root.
  - `_ticker_from_cache_path`: No function docstring.
  - `_timeframe_from_cache_path`: Determine the canonical timeframe from a cache file path.
  - `_all_cache_files`: Return de-duplicated daily + intraday cache candidate files for one root.
  - `rehydrate_cache_metadata`: Backfill metadata sidecars for existing cache files without rewriting price data.
  - `load_ibkr_data`: Scan a directory of IBKR-downloaded files (CSV or parquet).
  - `cache_universe`: Save all tickers in a data dict to the local cache.

### `data/provider_base.py` (12 lines)
- Intent: Shared provider protocol for pluggable data connectors.
- Classes:
  - `DataProvider`: No class docstring. Methods: `available`.
- Top-level functions: none

### `data/provider_registry.py` (49 lines)
- Intent: Provider registry for unified data-provider access (WRDS, Kalshi, ...).
- Classes: none
- Top-level functions:
  - `_wrds_factory`: No function docstring.
  - `_kalshi_factory`: No function docstring.
  - `get_provider`: No function docstring.
  - `list_providers`: No function docstring.
  - `register_provider`: No function docstring.

### `data/quality.py` (237 lines)
- Intent: Data quality checks for OHLCV time series.
- Classes:
  - `DataQualityReport`: No class docstring. Methods: `to_dict`.
- Top-level functions:
  - `assess_ohlcv_quality`: No function docstring.
  - `generate_quality_report`: Return a per-stock quality summary DataFrame.
  - `flag_degraded_stocks`: Return a list of tickers whose data quality is below threshold.

### `data/survivorship.py` (928 lines)
- Intent: Survivorship Bias Controls (Tasks 112-117)
- Classes:
  - `DelistingReason`: Reason for stock delisting. Methods: none.
  - `UniverseMember`: Task 112: Track a symbol's membership in a universe. Methods: `is_active_on`, `to_dict`.
  - `UniverseChange`: Task 114: Track a change to universe membership. Methods: `to_dict`.
  - `DelistingEvent`: Task 113: Track delisting event with proper returns. Methods: `to_dict`.
  - `SurvivorshipReport`: Task 117: Report comparing returns with/without survivorship adjustment. Methods: `to_dict`.
  - `UniverseHistoryTracker`: Task 112, 114, 115: Track historical universe membership. Methods: `__init__`, `_init_db`, `add_member`, `record_change`, `get_universe_on_date`, `get_changes_in_period`, `bulk_load_universe`, `clear_universe`.
  - `DelistingHandler`: Task 113, 116: Handle delisting events properly. Methods: `__init__`, `_init_db`, `record_delisting`, `preserve_price_history`, `get_dead_company_prices`, `get_delisting_event`, `get_delisting_return`, `is_delisted`, `get_all_delisted_symbols`.
  - `SurvivorshipBiasController`: Task 117: Main controller for survivorship bias analysis. Methods: `__init__`, `get_survivorship_free_universe`, `calculate_bias_impact`, `format_report`.
- Top-level functions:
  - `hydrate_universe_history_from_snapshots`: Build point-in-time universe intervals from snapshot rows.
  - `hydrate_sp500_history_from_wrds`: Pull historical S&P 500 snapshots from WRDS and hydrate local PIT DB.
  - `filter_panel_by_point_in_time_universe`: Filter MultiIndex panel rows by point-in-time universe membership.
  - `reconstruct_historical_universe`: Task 115: Quick function to reconstruct historical universe.
  - `calculate_survivorship_bias_impact`: Task 117: Quick function to calculate survivorship bias impact.

### `data/wrds_provider.py` (1584 lines)
- Intent: wrds_provider.py
- Classes:
  - `WRDSProvider`: WRDS data provider for the auto-discovery pipeline. Methods: `__init__`, `available`, `_query`, `_query_silent`, `get_sp500_universe`, `get_sp500_history`, `resolve_permno`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `get_optionmetrics_link`, `_nearest_iv`, `get_option_surface_features`, `get_fundamentals`, `get_earnings_surprises`, `get_institutional_ownership`, `get_taqmsec_ohlcv`, `query_options_volume`, `query_short_interest`, `query_insider_transactions`, `_permno_to_ticker`, `get_ohlcv`.
- Top-level functions:
  - `_sanitize_ticker_list`: Build a SQL-safe IN-clause string from ticker symbols.
  - `_sanitize_permno_list`: Build a SQL-safe IN-clause string from PERMNO values.
  - `_read_pgpass_password`: Read the WRDS password from ~/.pgpass so the wrds library doesn't
  - `_get_connection`: Get or create a cached WRDS connection. Returns None if unavailable.
  - `get_wrds_provider`: Get or create the default WRDSProvider singleton.
  - `wrds_available`: Quick check: is WRDS accessible?

## Package `features`
- Modules: 9
### `features/__init__.py` (0 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes: none
- Top-level functions: none

### `features/harx_spillovers.py` (242 lines)
- Intent: HARX Volatility Spillover features (Tier 6.1).
- Classes: none
- Top-level functions:
  - `_realized_volatility`: Rolling realized volatility (annualised std of returns).
  - `_ols_lstsq`: OLS via numpy lstsq.  Returns coefficient vector.
  - `compute_harx_spillovers`: Compute HARX cross-market volatility spillover features.

### `features/intraday.py` (193 lines)
- Intent: Intraday microstructure features from WRDS TAQmsec tick data.
- Classes: none
- Top-level functions:
  - `compute_intraday_features`: Compute intraday microstructure features for a single ticker on a date.

### `features/lob_features.py` (311 lines)
- Intent: Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).
- Classes: none
- Top-level functions:
  - `_inter_bar_durations`: Compute inter-bar durations in seconds from a DatetimeIndex.
  - `_estimate_poisson_lambda`: Estimate trade arrival rate (lambda) from inter-arrival durations.
  - `_signed_volume`: Approximate trade direction using candle body (close - open).
  - `compute_lob_features`: Compute Markov LOB proxy features for a single stock-day.
  - `compute_lob_features_batch`: Compute LOB features for multiple stock-days in batch.

### `features/macro.py` (243 lines)
- Intent: FRED macro indicator features for quant_engine.
- Classes:
  - `MacroFeatureProvider`: FRED API integration for macro indicator features. Methods: `__init__`, `_fetch_series_fredapi`, `_fetch_series_requests`, `_fetch_series`, `get_macro_features`.
- Top-level functions:
  - `_cache_key`: Generate a deterministic cache filename.

### `features/options_factors.py` (119 lines)
- Intent: Option surface factor construction from OptionMetrics-enriched daily panels.
- Classes: none
- Top-level functions:
  - `_pick_numeric`: No function docstring.
  - `_rolling_percentile_rank`: No function docstring.
  - `compute_option_surface_factors`: Compute minimal high-signal option surface features.
  - `compute_iv_shock_features`: Event-centric IV shock features (G3).

### `features/pipeline.py` (819 lines)
- Intent: Feature Pipeline — computes model features from OHLCV data.
- Classes:
  - `FeaturePipeline`: End-to-end feature computation pipeline. Methods: `__init__`, `compute`, `compute_universe`, `_load_benchmark_close`.
- Top-level functions:
  - `_build_indicator_set`: Instantiate all indicators with default parameters.
  - `_get_indicators`: No function docstring.
  - `compute_indicator_features`: Compute all indicator-based features as continuous columns.
  - `compute_raw_features`: Compute raw OHLCV-derived features (returns, volume, gaps, etc.).
  - `compute_har_volatility_features`: Compute HAR (Heterogeneous Autoregressive) realized volatility features.
  - `compute_multiscale_features`: Compute momentum, RSI, and volatility features at multiple time scales.
  - `compute_interaction_features`: Generate interaction features from pairs of continuous indicators.
  - `compute_targets`: Compute forward return targets for supervised learning.
  - `_winsorize_expanding`: Winsorize features using expanding-window quantiles (no look-ahead).

### `features/research_factors.py` (976 lines)
- Intent: Research-derived factor construction for quant_engine.
- Classes:
  - `ResearchFactorConfig`: Configuration for research-derived factor generation. Methods: none.
- Top-level functions:
  - `_rolling_zscore`: Causal rolling z-score.
  - `_safe_pct_change`: No function docstring.
  - `_required_ohlcv`: No function docstring.
  - `compute_order_flow_impact_factors`: Order-flow imbalance and price-impact proxies (Cont et al. inspired).
  - `compute_markov_queue_features`: Markov-style queue imbalance features (de Larrard style state framing).
  - `compute_time_series_momentum_factors`: Vol-scaled time-series momentum factors (Moskowitz/Ooi/Pedersen style).
  - `compute_vol_scaled_momentum`: Volatility-scaled time-series momentum enhancements.
  - `_rolling_levy_area`: Rolling Levy area for a 2D path of increments.
  - `compute_signature_path_features`: Signature-inspired path features for returns-volume trajectory.
  - `compute_vol_surface_factors`: Volatility term-structure factors inspired by implied-vol surface dynamics.
  - `compute_single_asset_research_factors`: Compute all single-asset research factors.
  - `_standardize_block`: Column-wise z-score with NaN-safe handling.
  - `_lagged_weight_matrix`: Build positive lagged correlation weights:
  - `compute_cross_asset_research_factors`: Compute cross-asset network momentum and volatility spillover factors.
  - `_dtw_distance_numpy`: Pure numpy DTW distance computation using dynamic programming.
  - `_dtw_avg_lag_from_path`: Extract average lag from DTW alignment path.
  - `compute_dtw_lead_lag`: DTW-based lead-lag detection across a universe of assets.
  - `_numpy_order2_signature`: Pure numpy computation of truncated order-2 path signature for a 2D path
  - `compute_path_signatures`: Compute truncated path signatures of (price, volume) paths.

### `features/wave_flow.py` (144 lines)
- Intent: Wave-Flow Decomposition for quant_engine.
- Classes: none
- Top-level functions:
  - `compute_wave_flow_decomposition`: Decompose the return series into flow (secular trend) and wave (oscillatory)

## Package `indicators`
- Modules: 2
### `indicators/__init__.py` (57 lines)
- Intent: Quant Engine Indicators — self-contained copy of the technical indicator library.
- Classes: none
- Top-level functions: none

### `indicators/indicators.py` (2635 lines)
- Intent: Technical Indicator Library
- Classes:
  - `Indicator`: Base class for all indicators. Methods: `name`, `calculate`.
  - `ATR`: Average True Range - measures volatility. Methods: `__init__`, `name`, `calculate`.
  - `NATR`: Normalized ATR - ATR as percentage of close price. Methods: `__init__`, `name`, `calculate`.
  - `BollingerBandWidth`: Bollinger Band Width - measures volatility squeeze. Methods: `__init__`, `name`, `calculate`.
  - `HistoricalVolatility`: Historical volatility (standard deviation of returns). Methods: `__init__`, `name`, `calculate`.
  - `RSI`: Relative Strength Index. Methods: `__init__`, `name`, `calculate`.
  - `MACD`: MACD Line (difference between fast and slow EMA). Methods: `__init__`, `name`, `calculate`.
  - `MACDSignal`: MACD Signal Line. Methods: `__init__`, `name`, `calculate`.
  - `MACDHistogram`: MACD Histogram (MACD - Signal). Methods: `__init__`, `name`, `calculate`.
  - `ROC`: Rate of Change. Methods: `__init__`, `name`, `calculate`.
  - `Stochastic`: Stochastic %K. Methods: `__init__`, `name`, `calculate`.
  - `StochasticD`: Stochastic %D (smoothed %K). Methods: `__init__`, `name`, `calculate`.
  - `WilliamsR`: Williams %R. Methods: `__init__`, `name`, `calculate`.
  - `CCI`: Commodity Channel Index. Methods: `__init__`, `name`, `calculate`.
  - `SMA`: Simple Moving Average. Methods: `__init__`, `name`, `calculate`.
  - `EMA`: Exponential Moving Average. Methods: `__init__`, `name`, `calculate`.
  - `PriceVsSMA`: Price distance from SMA (as percentage). Methods: `__init__`, `name`, `calculate`.
  - `SMASlope`: Slope of SMA (rate of change). Methods: `__init__`, `name`, `calculate`.
  - `ADX`: Average Directional Index - trend strength. Methods: `__init__`, `name`, `calculate`.
  - `Aroon`: Aroon Oscillator. Methods: `__init__`, `name`, `calculate`.
  - `VolumeRatio`: Current volume vs average volume. Methods: `__init__`, `name`, `calculate`.
  - `OBV`: On-Balance Volume. Methods: `name`, `calculate`.
  - `OBVSlope`: OBV rate of change. Methods: `__init__`, `name`, `calculate`.
  - `MFI`: Money Flow Index. Methods: `__init__`, `name`, `calculate`.
  - `HigherHighs`: Count of higher highs in lookback period. Methods: `__init__`, `name`, `calculate`.
  - `LowerLows`: Count of lower lows in lookback period. Methods: `__init__`, `name`, `calculate`.
  - `CandleBody`: Candle body size as percentage of range. Methods: `name`, `calculate`.
  - `CandleDirection`: Candle direction streak (positive = up candles, negative = down). Methods: `__init__`, `name`, `calculate`.
  - `GapPercent`: Gap from previous close as percentage. Methods: `name`, `calculate`.
  - `DistanceFromHigh`: Distance from N-period high as percentage. Methods: `__init__`, `name`, `calculate`.
  - `DistanceFromLow`: Distance from N-period low as percentage. Methods: `__init__`, `name`, `calculate`.
  - `PricePercentile`: Current price percentile within N-period range. Methods: `__init__`, `name`, `calculate`.
  - `BBWidthPercentile`: Bollinger Band Width Percentile - identifies squeeze conditions. Methods: `__init__`, `name`, `calculate`.
  - `NATRPercentile`: NATR Percentile - where current volatility sits vs history. Methods: `__init__`, `name`, `calculate`.
  - `VolatilitySqueeze`: Volatility Squeeze indicator - BB inside Keltner Channel. Methods: `__init__`, `name`, `calculate`.
  - `RVOL`: Relative Volume - current volume vs same time period average. Methods: `__init__`, `name`, `calculate`.
  - `NetVolumeTrend`: Net Volume Trend - accumulation/distribution pressure. Methods: `__init__`, `name`, `calculate`.
  - `VolumeForce`: Volume Force Index - measures buying/selling pressure. Methods: `__init__`, `name`, `calculate`.
  - `AccumulationDistribution`: Accumulation/Distribution Line slope. Methods: `__init__`, `name`, `calculate`.
  - `EMAAlignment`: EMA Alignment - checks if EMAs are properly stacked. Methods: `__init__`, `name`, `calculate`.
  - `TrendStrength`: Combined trend strength using multiple factors. Methods: `__init__`, `name`, `calculate`.
  - `PriceVsEMAStack`: Price position relative to EMA stack. Methods: `name`, `calculate`.
  - `PivotHigh`: Pivot High breakout - price breaks above N-bar high. Methods: `__init__`, `name`, `calculate`.
  - `PivotLow`: Pivot Low breakdown - price breaks below N-bar low. Methods: `__init__`, `name`, `calculate`.
  - `NBarHighBreak`: Simple N-bar high breakout. Methods: `__init__`, `name`, `calculate`.
  - `NBarLowBreak`: Simple N-bar low breakdown. Methods: `__init__`, `name`, `calculate`.
  - `RangeBreakout`: Range Breakout - price breaks out of N-day range. Methods: `__init__`, `name`, `calculate`.
  - `ATRTrailingStop`: Distance from ATR trailing stop. Methods: `__init__`, `name`, `calculate`.
  - `ATRChannel`: Position within ATR channel. Methods: `__init__`, `name`, `calculate`.
  - `RiskPerATR`: Recent price range in ATR units. Methods: `__init__`, `name`, `calculate`.
  - `MarketRegime`: Market regime based on price action. Methods: `__init__`, `name`, `calculate`.
  - `VolatilityRegime`: Volatility regime classification. Methods: `__init__`, `name`, `calculate`.
  - `VWAP`: Volume Weighted Average Price - rolling calculation. Methods: `__init__`, `name`, `calculate`.
  - `PriceVsVWAP`: Price distance from VWAP as percentage. Methods: `__init__`, `name`, `calculate`.
  - `VWAPBands`: VWAP Standard Deviation Bands. Methods: `__init__`, `name`, `calculate`.
  - `AnchoredVWAP`: Anchored VWAP - VWAP calculated from N days ago. Methods: `__init__`, `name`, `calculate`.
  - `PriceVsAnchoredVWAP`: Price distance from Anchored VWAP. Methods: `__init__`, `name`, `calculate`.
  - `MultiVWAPPosition`: Position relative to multiple VWAP anchors. Methods: `name`, `calculate`.
  - `ValueAreaHigh`: Value Area High approximation. Methods: `__init__`, `name`, `calculate`.
  - `ValueAreaLow`: Value Area Low approximation. Methods: `__init__`, `name`, `calculate`.
  - `POC`: Point of Control approximation. Methods: `__init__`, `name`, `calculate`.
  - `PriceVsPOC`: Price distance from Point of Control. Methods: `__init__`, `name`, `calculate`.
  - `ValueAreaPosition`: Position within Value Area. Methods: `__init__`, `name`, `calculate`.
  - `AboveValueArea`: Binary: 1 if price above VAH, 0 otherwise. Methods: `__init__`, `name`, `calculate`.
  - `BelowValueArea`: Binary: 1 if price below VAL, 0 otherwise. Methods: `__init__`, `name`, `calculate`.
  - `Beast666Proximity`: Beast 666 Proximity Score (0-100). Methods: `__init__`, `name`, `calculate`.
  - `Beast666Distance`: Signed percent distance from the nearest 666 level. Methods: `__init__`, `name`, `calculate`.
  - `ParkinsonVolatility`: Parkinson range-based volatility estimator. More efficient than close-to-close. Methods: `__init__`, `name`, `calculate`.
  - `GarmanKlassVolatility`: Garman-Klass OHLC volatility estimator. ~8x more efficient than close-to-close. Methods: `__init__`, `name`, `calculate`.
  - `YangZhangVolatility`: Yang-Zhang volatility combining overnight and Rogers-Satchell intraday components. Methods: `__init__`, `name`, `calculate`.
  - `VolatilityCone`: Percentile rank of current realized vol vs its historical distribution. Methods: `__init__`, `name`, `calculate`.
  - `VolOfVol`: Volatility of volatility - rolling std of rolling volatility. Methods: `__init__`, `name`, `calculate`.
  - `GARCHVolatility`: Simplified GARCH(1,1) volatility with fixed parameters. Methods: `__init__`, `name`, `calculate`.
  - `VolTermStructure`: Ratio of short-term to long-term realized vol. >1 = backwardation (fear). Methods: `__init__`, `name`, `calculate`.
  - `HurstExponent`: Hurst exponent via R/S analysis. H>0.5 trending, H<0.5 mean-reverting. Methods: `__init__`, `name`, `calculate`.
  - `MeanReversionHalfLife`: Ornstein-Uhlenbeck half-life via OLS. Lower = faster mean reversion. Methods: `__init__`, `name`, `calculate`.
  - `ZScore`: Z-Score: standardized deviation from rolling mean. Methods: `__init__`, `name`, `calculate`.
  - `VarianceRatio`: Lo-MacKinlay variance ratio. VR>1 = trending, VR<1 = mean-reverting. Methods: `__init__`, `name`, `calculate`.
  - `Autocorrelation`: Serial correlation of returns at lag k. Positive = momentum, negative = mean-reversion. Methods: `__init__`, `name`, `calculate`.
  - `KalmanTrend`: 1D Kalman filter for price trend extraction. Methods: `__init__`, `name`, `calculate`.
  - `ShannonEntropy`: Shannon entropy of return distribution. High = uncertain, low = predictable. Methods: `__init__`, `name`, `calculate`.
  - `ApproximateEntropy`: Approximate Entropy (ApEn). Low = regular/predictable, high = complex/random. Methods: `__init__`, `name`, `calculate`.
  - `AmihudIlliquidity`: Amihud illiquidity ratio: |return| / dollar_volume. Higher = less liquid. Methods: `__init__`, `name`, `calculate`.
  - `KyleLambda`: Kyle's lambda price impact coefficient via rolling regression. Methods: `__init__`, `name`, `calculate`.
  - `RollSpread`: Roll's implied bid-ask spread in basis points. Methods: `__init__`, `name`, `calculate`.
  - `FractalDimension`: Higuchi fractal dimension. D~1 = smooth/trending, D~2 = rough/noisy. Methods: `__init__`, `name`, `calculate`.
  - `DFA`: Detrended Fluctuation Analysis. alpha>0.5 = persistent, alpha<0.5 = anti-persistent. Methods: `__init__`, `name`, `calculate`.
  - `DominantCycle`: FFT-based dominant cycle period in bars. Methods: `__init__`, `name`, `calculate`.
  - `ReturnSkewness`: Rolling skewness of returns. Negative = left tail risk. Methods: `__init__`, `name`, `calculate`.
  - `ReturnKurtosis`: Rolling excess kurtosis. High = fat tails (tail risk). Methods: `__init__`, `name`, `calculate`.
  - `CUSUMDetector`: CUSUM change-point detection. Output = bars since last regime change / period. Methods: `__init__`, `name`, `calculate`.
  - `RegimePersistence`: Consecutive bars in the same trend regime (price vs SMA). Methods: `__init__`, `name`, `calculate`.
- Top-level functions:
  - `get_all_indicators`: Return dictionary of all indicator classes.
  - `create_indicator`: Create an indicator by name with given parameters.

## Package `kalshi`
- Modules: 25
### `kalshi/__init__.py` (58 lines)
- Intent: Kalshi vertical for intraday event-market research.
- Classes: none
- Top-level functions: none

### `kalshi/client.py` (630 lines)
- Intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- Classes:
  - `RetryPolicy`: No class docstring. Methods: none.
  - `RateLimitPolicy`: No class docstring. Methods: none.
  - `RequestLimiter`: Lightweight token-bucket limiter with runtime limit updates. Methods: `__init__`, `_refill`, `acquire`, `update_rate`, `update_from_account_limits`.
  - `KalshiSigner`: Signs Kalshi requests using RSA-PSS SHA256. Methods: `__init__`, `available`, `_canonical_path`, `_load_private_key`, `_sign_with_cryptography`, `_sign_with_openssl`, `sign`.
  - `KalshiClient`: Kalshi HTTP wrapper with: Methods: `__init__`, `available`, `_join_url`, `_auth_headers`, `_request_with_retries`, `_request`, `get`, `paginate`, `get_account_limits`, `fetch_historical_cutoff`, `server_time_utc`, `clock_skew_seconds`, `list_markets`, `list_contracts`, `list_trades`, `list_quotes`.
- Top-level functions:
  - `_normalize_env`: No function docstring.

### `kalshi/disagreement.py` (112 lines)
- Intent: Cross-market disagreement engine for Kalshi event features.
- Classes:
  - `DisagreementSignals`: No class docstring. Methods: none.
- Top-level functions:
  - `compute_disagreement`: Compute cross-market disagreement signals.
  - `disagreement_as_feature_dict`: Convert disagreement signals to a flat dict for feature panel merging.

### `kalshi/distribution.py` (917 lines)
- Intent: Contract -> probability distribution builder for Kalshi markets.
- Classes:
  - `DistributionConfig`: No class docstring. Methods: none.
  - `DirectionResult`: Result of threshold direction resolution with confidence metadata. Methods: none.
  - `BinValidationResult`: Result of bin overlap/gap/ordering validation. Methods: none.
- Top-level functions:
  - `_is_tz_aware_datetime`: No function docstring.
  - `_to_utc_timestamp`: No function docstring.
  - `_prob_from_mid`: No function docstring.
  - `_entropy`: No function docstring.
  - `_isotonic_nonincreasing`: Pool-adjacent-violators for nonincreasing constraints.
  - `_isotonic_nondecreasing`: No function docstring.
  - `_resolve_threshold_direction`: Resolve threshold contract semantics:
  - `_resolve_threshold_direction_with_confidence`: Resolve threshold contract semantics with confidence scoring.
  - `_validate_bins`: Validate bin structure for non-overlapping, ordered coverage.
  - `_tail_thresholds`: No function docstring.
  - `_latest_quotes_asof`: No function docstring.
  - `_normalize_mass`: No function docstring.
  - `_moments`: No function docstring.
  - `_cdf_from_pmf`: No function docstring.
  - `_pmf_on_grid`: No function docstring.
  - `_distribution_distances`: No function docstring.
  - `_tail_probs_from_mass`: No function docstring.
  - `_tail_probs_from_threshold_curve`: No function docstring.
  - `_estimate_liquidity_proxy`: Estimate a stable liquidity proxy from recent quote stream.
  - `build_distribution_snapshot`: No function docstring.
  - `_lag_slug`: No function docstring.
  - `_add_distance_features`: No function docstring.
  - `build_distribution_panel`: Build market-level distribution snapshots across times.

### `kalshi/events.py` (511 lines)
- Intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- Classes:
  - `EventTimestampMeta`: Authoritative event timestamp metadata (D2). Methods: none.
  - `EventFeatureConfig`: No class docstring. Methods: none.
- Top-level functions:
  - `_to_utc_ts`: No function docstring.
  - `_ensure_asof_before_release`: No function docstring.
  - `asof_join`: Strict backward as-of join (no forward-peeking).
  - `build_event_snapshot_grid`: No function docstring.
  - `_merge_event_market_map`: No function docstring.
  - `_add_revision_speed_features`: No function docstring.
  - `add_reference_disagreement_features`: Optional cross-market disagreement block via strict backward as-of join.
  - `build_event_feature_panel`: Build event-centric panel indexed by (event_id, asof_ts).
  - `build_asset_time_feature_panel`: Optional continuous panel keyed by (asset_id, ts) with strict as-of joins.
  - `build_event_labels`: Build event outcome labels with explicit as-of awareness.
  - `build_asset_response_labels`: Event-to-asset response labels with realistic execution windows.

### `kalshi/mapping_store.py` (70 lines)
- Intent: Versioned event-to-market mapping persistence.
- Classes:
  - `EventMarketMappingRecord`: No class docstring. Methods: none.
  - `EventMarketMappingStore`: No class docstring. Methods: `__init__`, `upsert`, `asof`, `current_version`, `assert_consistent_mapping_version`.
- Top-level functions: none

### `kalshi/microstructure.py` (126 lines)
- Intent: Market microstructure diagnostics for Kalshi event markets.
- Classes:
  - `MicrostructureDiagnostics`: No class docstring. Methods: none.
- Top-level functions:
  - `compute_microstructure`: Compute microstructure diagnostics from a quote panel.
  - `microstructure_as_feature_dict`: Convert diagnostics to a flat dict for feature panel merging.

### `kalshi/options.py` (144 lines)
- Intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- Classes: none
- Top-level functions:
  - `_to_utc_ts`: No function docstring.
  - `build_options_reference_panel`: Build a normalized options reference panel with:
  - `add_options_disagreement_features`: Strict backward as-of join of options reference features into event panel.

### `kalshi/pipeline.py` (158 lines)
- Intent: Orchestration helpers for the Kalshi event-market vertical.
- Classes:
  - `KalshiPipeline`: No class docstring. Methods: `from_store`, `sync_reference`, `sync_intraday_quotes`, `build_distributions`, `build_event_features`, `run_walkforward`, `evaluate_walkforward_contract`, `evaluate_event_promotion`.
- Top-level functions: none

### `kalshi/promotion.py` (176 lines)
- Intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- Classes:
  - `EventPromotionConfig`: No class docstring. Methods: none.
- Top-level functions:
  - `_to_backtest_result`: No function docstring.
  - `evaluate_event_promotion`: Evaluate Kalshi event strategy promotion from walk-forward outputs.

### `kalshi/provider.py` (628 lines)
- Intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- Classes:
  - `KalshiProvider`: Provider interface similar to WRDSProvider, but for event-market data. Methods: `__init__`, `available`, `sync_account_limits`, `refresh_historical_cutoff`, `sync_market_catalog`, `sync_contracts`, `sync_quotes`, `get_markets`, `get_contracts`, `get_quotes`, `get_event_market_map_asof`, `get_macro_events`, `get_event_outcomes`, `compute_and_store_distributions`, `materialize_daily_health_report`, `get_daily_health_report`, `store_clock_check`.
- Top-level functions:
  - `_to_iso_utc`: No function docstring.
  - `_safe_hash_text`: No function docstring.
  - `_asof_date`: No function docstring.

### `kalshi/quality.py` (203 lines)
- Intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- Classes:
  - `QualityDimensions`: No class docstring. Methods: none.
  - `StalePolicy`: No class docstring. Methods: none.
- Top-level functions:
  - `_finite`: No function docstring.
  - `dynamic_stale_cutoff_minutes`: Dynamic stale-cutoff schedule:
  - `compute_quality_dimensions`: Multi-dimensional quality model for distribution snapshots.
  - `passes_hard_gates`: Hard validity gates (C1).  Must-pass criteria — failing any gate means
  - `quality_as_feature_dict`: Expose soft quality dimensions as separate learnable feature columns (C2).

### `kalshi/regimes.py` (141 lines)
- Intent: Regime tagging for Kalshi event strategies.
- Classes:
  - `EventRegimeTag`: No class docstring. Methods: none.
- Top-level functions:
  - `classify_inflation_regime`: Classify inflation regime from CPI year-over-year.
  - `classify_policy_regime`: Classify monetary policy regime from Fed funds rate changes.
  - `classify_vol_regime`: Classify volatility regime from VIX level.
  - `tag_event_regime`: Tag an event with macro regime classifications.
  - `evaluate_strategy_by_regime`: Evaluate strategy performance breakdown by regime.
  - `regime_stability_score`: Score strategy stability across regimes (0-1).

### `kalshi/router.py` (95 lines)
- Intent: Routing helpers for live vs historical Kalshi endpoints.
- Classes:
  - `RouteDecision`: No class docstring. Methods: none.
  - `KalshiDataRouter`: Chooses live vs historical endpoint roots by cutoff timestamp. Methods: `__init__`, `_to_utc_ts`, `update_cutoff`, `_extract_end_ts`, `_clean_path`, `resolve`.
- Top-level functions: none

### `kalshi/storage.py` (623 lines)
- Intent: Event-time storage layer for Kalshi + macro event research.
- Classes:
  - `EventTimeStore`: Intraday/event-time storage with a stable schema. Methods: `__init__`, `_execute`, `_executemany`, `_table_columns`, `_norm_ts`, `_clean_value`, `_insert_or_replace`, `init_schema`, `upsert_markets`, `upsert_contracts`, `append_quotes`, `upsert_macro_events`, `upsert_event_outcomes`, `upsert_event_outcomes_first_print`, `upsert_event_outcomes_revised`, `upsert_distributions`, `upsert_event_market_map_versions`, `append_market_specs`, `append_contract_specs`, `upsert_data_provenance`, `upsert_coverage_diagnostics`, `upsert_ingestion_logs`, `upsert_daily_health_reports`, `upsert_ingestion_checkpoints`, `get_ingestion_checkpoint`, `get_event_market_map_asof`, `query_df`.
- Top-level functions: none

### `kalshi/tests/__init__.py` (1 lines)
- Intent: Kalshi package-local tests.
- Classes: none
- Top-level functions: none

### `kalshi/tests/test_bin_validity.py` (105 lines)
- Intent: Bin overlap/gap detection test (Instructions I.3).
- Classes:
  - `BinValidityTests`: Tests for bin overlap/gap/ordering validation. Methods: `test_clean_bins_valid`, `test_overlapping_bins_detected`, `test_gapped_bins_detected`, `test_inverted_bin_detected`, `test_single_bin_valid`, `test_missing_columns_valid`, `test_empty_dataframe_valid`, `test_unordered_bins_detected`, `test_severe_overlap`.
- Top-level functions: none

### `kalshi/tests/test_distribution.py` (36 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `DistributionLocalTests`: No class docstring. Methods: `test_bin_distribution_probability_mass_is_normalized`.
- Top-level functions: none

### `kalshi/tests/test_leakage.py` (41 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `LeakageLocalTests`: No class docstring. Methods: `test_feature_rows_strictly_pre_release`.
- Top-level functions: none

### `kalshi/tests/test_no_leakage.py` (117 lines)
- Intent: No-leakage test at panel level (Instructions I.4).
- Classes:
  - `NoLeakageTests`: Panel-level look-ahead bias detection. Methods: `_build_synthetic_panel`, `test_all_asof_before_release`, `test_single_event_no_leakage`.
- Top-level functions: none

### `kalshi/tests/test_signature_kat.py` (141 lines)
- Intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- Classes:
  - `SignatureKATTests`: Known-answer tests for Kalshi request signing. Methods: `_skip_if_no_crypto`, `test_sign_produces_valid_base64`, `test_sign_deterministic_message_format`, `test_sign_verifies_with_public_key`, `test_canonical_path_normalization`.
- Top-level functions: none

### `kalshi/tests/test_stale_quotes.py` (152 lines)
- Intent: Stale quote cutoff test (Instructions I.5).
- Classes:
  - `StaleQuoteCutoffTests`: Tests for dynamic stale-cutoff schedule. Methods: `test_near_event_tight_cutoff`, `test_far_event_loose_cutoff`, `test_midpoint_interpolation`, `test_cutoff_monotonically_increases_with_distance`, `test_cpi_market_type_multiplier`, `test_fomc_market_type_multiplier`, `test_low_liquidity_widens_cutoff`, `test_high_liquidity_tightens_cutoff`, `test_none_time_to_event_uses_base`, `test_cutoff_clamped_to_bounds`.
- Top-level functions: none

### `kalshi/tests/test_threshold_direction.py` (126 lines)
- Intent: Threshold direction correctness test (Instructions I.2).
- Classes:
  - `ThresholdDirectionTests`: Tests for threshold direction resolution. Methods: `test_explicit_ge_direction`, `test_explicit_le_direction`, `test_explicit_gte_alias`, `test_explicit_lte_alias`, `test_explicit_ge_symbol`, `test_explicit_le_symbol`, `test_payout_structure_above`, `test_payout_structure_below`, `test_rules_text_greater_than`, `test_rules_text_less_than`, `test_rules_text_above`, `test_rules_text_below`, `test_title_guess_or_higher`, `test_title_guess_or_lower`, `test_no_direction_signal`, `test_empty_row`, `test_legacy_resolve_returns_string`.
- Top-level functions: none

### `kalshi/tests/test_walkforward_purge.py` (159 lines)
- Intent: Walk-forward purge/embargo test (Instructions I.6).
- Classes:
  - `WalkForwardPurgeTests`: Tests that walk-forward purge/embargo prevents data leakage. Methods: `_build_synthetic_data`, `test_no_train_events_in_purge_window`, `test_event_type_aware_purge`, `test_embargo_removes_adjacent_events`, `test_trial_counting`.
- Top-level functions: none

### `kalshi/walkforward.py` (477 lines)
- Intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- Classes:
  - `EventWalkForwardConfig`: No class docstring. Methods: none.
  - `EventWalkForwardFold`: No class docstring. Methods: none.
  - `EventWalkForwardResult`: No class docstring. Methods: `wf_oos_corr`, `wf_positive_fold_fraction`, `wf_is_oos_gap`, `worst_event_loss`, `to_metrics`.
- Top-level functions:
  - `_bootstrap_mean_ci`: No function docstring.
  - `_event_regime_stability`: No function docstring.
  - `evaluate_event_contract_metrics`: Advanced validation contract metrics for event strategies:
  - `_corr`: No function docstring.
  - `_fit_ridge`: No function docstring.
  - `_predict`: No function docstring.
  - `_prepare_panel`: No function docstring.
  - `run_event_walkforward`: No function docstring.

## Package `models`
- Modules: 13
### `models/__init__.py` (20 lines)
- Intent: Models subpackage — training, prediction, versioning, and retraining triggers.
- Classes: none
- Top-level functions: none

### `models/calibration.py` (216 lines)
- Intent: Confidence Calibration --- Platt scaling and isotonic regression.
- Classes:
  - `_LinearRescaler`: Maps raw scores to [0, 1] via min-max linear rescaling. Methods: `__init__`, `fit`, `transform`.
  - `ConfidenceCalibrator`: Post-hoc confidence calibration via Platt scaling or isotonic regression. Methods: `__init__`, `fit`, `_fit_sklearn`, `transform`, `fit_transform`, `is_fitted`, `backend`, `__repr__`.
- Top-level functions: none

### `models/cross_sectional.py` (136 lines)
- Intent: Cross-Sectional Ranking Model — rank stocks relative to peers at each date.
- Classes: none
- Top-level functions:
  - `cross_sectional_rank`: Rank stocks cross-sectionally by predicted return at each date.

### `models/feature_stability.py` (311 lines)
- Intent: Feature Stability Monitoring — tracks feature importance rankings across
- Classes:
  - `StabilityReport`: Summary returned by :meth:`FeatureStabilityTracker.check_stability`. Methods: `to_dict`.
  - `FeatureStabilityTracker`: Record and compare feature importance rankings over training cycles. Methods: `__init__`, `_load`, `_save`, `record_importance`, `_spearman_rank_correlation`, `check_stability`.
- Top-level functions: none

### `models/governance.py` (108 lines)
- Intent: Champion/challenger governance for model versions.
- Classes:
  - `ChampionRecord`: No class docstring. Methods: `to_dict`.
  - `ModelGovernance`: Maintains champion model per horizon and promotes challengers if better. Methods: `__init__`, `_load`, `_save`, `_score`, `get_champion_version`, `evaluate_and_update`.
- Top-level functions: none

### `models/iv/__init__.py` (31 lines)
- Intent: Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface.
- Classes: none
- Top-level functions: none

### `models/iv/models.py` (928 lines)
- Intent: Implied Volatility Surface Models.
- Classes:
  - `OptionType`: No class docstring. Methods: none.
  - `Greeks`: Option Greeks container. Methods: none.
  - `HestonParams`: Heston model parameters. Methods: `validate`.
  - `SVIParams`: Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)). Methods: none.
  - `BlackScholes`: Black-Scholes option pricing and analytics. Methods: `price`, `greeks`, `implied_vol`, `iv_surface`.
  - `HestonModel`: Heston (1993) stochastic volatility model. Methods: `__init__`, `characteristic_function`, `price`, `implied_vol`, `iv_surface`, `calibrate`.
  - `SVIModel`: SVI (Stochastic Volatility Inspired) implied variance parameterization. Methods: `__init__`, `total_variance`, `implied_vol`, `iv_surface`, `smile`, `calibrate`, `check_no_butterfly_arbitrage`.
  - `ArbitrageFreeSVIBuilder`: Arbitrage-aware SVI surface builder. Methods: `__init__`, `_svi_total_variance`, `_initial_guess`, `_vega_spread_weights`, `_slice_objective`, `fit_slice`, `enforce_calendar_monotonicity`, `interpolate_total_variance`, `build_surface`.
  - `IVPoint`: Single implied-volatility observation. Methods: none.
  - `IVSurface`: Store and interpolate an implied-volatility surface. Methods: `__init__`, `add_point`, `add_slice`, `add_surface`, `n_points`, `_log_moneyness`, `_build_interpolator`, `get_iv`, `get_smile`, `decompose`, `decompose_surface`.
- Top-level functions:
  - `generate_synthetic_market_surface`: Generate a realistic synthetic market IV surface for demonstration.

### `models/neural_net.py` (197 lines)
- Intent: Tabular Neural Network — feedforward network for tabular financial data.
- Classes:
  - `TabularNet`: Feedforward network for tabular financial data. Methods: `__init__`, `_build_model`, `fit`, `predict`, `feature_importances_`.
- Top-level functions: none

### `models/predictor.py` (375 lines)
- Intent: Model Predictor — loads trained ensemble and generates predictions.
- Classes:
  - `EnsemblePredictor`: Loads a trained regime-conditional ensemble and generates predictions. Methods: `__init__`, `_resolve_model_dir`, `_load`, `predict`, `predict_single`.
- Top-level functions:
  - `_prepare_features`: Align, impute, and return features matching expected column order.

### `models/retrain_trigger.py` (296 lines)
- Intent: ML Retraining Trigger Logic
- Classes:
  - `RetrainTrigger`: Determines when ML model should be retrained. Methods: `__init__`, `_load_metadata`, `_save_metadata`, `add_trade_result`, `check`, `record_retraining`, `status`.
- Top-level functions: none

### `models/trainer.py` (1340 lines)
- Intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- Classes:
  - `IdentityScaler`: No-op scaler that passes data through unchanged. Methods: `fit`, `transform`, `fit_transform`, `inverse_transform`.
  - `DiverseEnsemble`: Lightweight ensemble wrapper that combines predictions from multiple Methods: `__init__`, `_aggregate_feature_importances`, `predict`.
  - `TrainResult`: Result of training a single model. Methods: none.
  - `EnsembleResult`: Result of training the full regime-conditional ensemble. Methods: none.
  - `ModelTrainer`: Trains a regime-conditional gradient boosting ensemble for Methods: `__init__`, `_spearmanr`, `_require_sklearn`, `_extract_dates`, `_sort_panel_by_time`, `_temporal_holdout_masks`, `_date_purged_folds`, `_prune_correlated_features`, `_select_features`, `train_ensemble`, `_train_single`, `_train_diverse_ensemble`, `_optimize_ensemble_weights`, `_clone_model`, `_make_model`, `_save`, `_print_summary`.
- Top-level functions: none

### `models/versioning.py` (204 lines)
- Intent: Model Versioning — timestamped model directories with registry.
- Classes:
  - `ModelVersion`: Metadata for a single model version. Methods: `to_dict`, `from_dict`.
  - `ModelRegistry`: Manages versioned model storage and retrieval. Methods: `__init__`, `_load_registry`, `_save_registry`, `latest_version_id`, `get_latest`, `get_version`, `get_version_dir`, `get_latest_dir`, `list_versions`, `create_version_dir`, `register_version`, `rollback`, `prune_old`, `has_versions`.
- Top-level functions: none

### `models/walk_forward.py` (235 lines)
- Intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- Classes: none
- Top-level functions:
  - `_spearmanr`: Spearman rank correlation (scipy optional).
  - `_expanding_walk_forward_folds`: Generate expanding-window walk-forward folds using unique dates.
  - `_extract_dates`: Return row-aligned timestamps from an index (supports panel MultiIndex).
  - `walk_forward_select`: Select the best model configuration via walk-forward cross-validation.

## Package `regime`
- Modules: 4
### `regime/__init__.py` (14 lines)
- Intent: Regime modeling components.
- Classes: none
- Top-level functions: none

### `regime/correlation.py` (212 lines)
- Intent: Correlation Regime Detection (NEW 11).
- Classes:
  - `CorrelationRegimeDetector`: Detect regime changes in pairwise correlation structure. Methods: `__init__`, `compute_rolling_correlation`, `detect_correlation_spike`, `get_correlation_features`.
- Top-level functions: none

### `regime/detector.py` (292 lines)
- Intent: Regime detector with two engines:
- Classes:
  - `RegimeOutput`: No class docstring. Methods: none.
  - `RegimeDetector`: Classifies market regime at each bar using either rules or HMM. Methods: `__init__`, `detect`, `_rule_detect`, `_hmm_detect`, `detect_with_confidence`, `detect_full`, `regime_features`, `_get_col`.
- Top-level functions:
  - `detect_regimes_batch`: Shared regime detection across multiple PERMNOs.

### `regime/hmm.py` (501 lines)
- Intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- Classes:
  - `HMMFitResult`: No class docstring. Methods: none.
  - `GaussianHMM`: Gaussian HMM using EM (Baum-Welch). Methods: `__init__`, `_ensure_positive_definite`, `_init_params`, `_log_emission`, `_forward_backward`, `viterbi`, `_smooth_duration`, `fit`, `predict_proba`.
- Top-level functions:
  - `_logsumexp`: No function docstring.
  - `select_hmm_states_bic`: Select the optimal number of HMM states using the Bayesian Information Criterion.
  - `build_hmm_observation_matrix`: Build a robust, low-dimensional observation matrix for regime inference.
  - `map_raw_states_to_regimes`: Map unlabeled HMM states -> semantic regimes used by the system.

## Package `risk`
- Modules: 11
### `risk/__init__.py` (42 lines)
- Intent: Risk Management Module — Renaissance-grade portfolio risk controls.
- Classes: none
- Top-level functions: none

### `risk/attribution.py` (266 lines)
- Intent: Performance Attribution --- decompose portfolio returns into market, factor, and alpha.
- Classes: none
- Top-level functions:
  - `_estimate_beta`: OLS beta of portfolio vs benchmark.
  - `_estimate_factor_loadings`: Multivariate OLS regression of excess returns on factor returns.
  - `decompose_returns`: Decompose portfolio returns into market, factor, and alpha components.
  - `compute_attribution_report`: Produce an extended attribution summary with risk-adjusted metrics.

### `risk/covariance.py` (244 lines)
- Intent: Covariance estimation utilities for portfolio risk controls.
- Classes:
  - `CovarianceEstimate`: No class docstring. Methods: none.
  - `CovarianceEstimator`: Estimate a robust covariance matrix for asset returns. Methods: `__init__`, `estimate`, `portfolio_volatility`, `_estimate_values`.
- Top-level functions:
  - `compute_regime_covariance`: Compute separate covariance matrices for each market regime.
  - `get_regime_covariance`: Return the covariance matrix for *current_regime*.

### `risk/drawdown.py` (233 lines)
- Intent: Drawdown Controller — circuit breakers and recovery protocols.
- Classes:
  - `DrawdownState`: No class docstring. Methods: none.
  - `DrawdownStatus`: Current drawdown state and action directives. Methods: none.
  - `DrawdownController`: Multi-tier drawdown protection with circuit breakers. Methods: `__init__`, `update`, `_compute_actions`, `reset`, `get_summary`.
- Top-level functions: none

### `risk/factor_portfolio.py` (220 lines)
- Intent: Factor-Based Portfolio Construction — factor decomposition and exposure analysis.
- Classes: none
- Top-level functions:
  - `compute_factor_exposures`: Estimate factor betas for each asset via OLS regression.
  - `compute_residual_returns`: Strip out systematic factor exposure, returning idiosyncratic returns.

### `risk/metrics.py` (251 lines)
- Intent: Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.
- Classes:
  - `RiskReport`: Comprehensive risk metrics report. Methods: none.
  - `RiskMetrics`: Computes comprehensive risk metrics from trade returns and equity curves. Methods: `__init__`, `compute_full_report`, `_drawdown_analytics`, `_drawdown_analytics_array`, `_compute_mae_mfe`, `_empty_report`, `print_report`.
- Top-level functions: none

### `risk/portfolio_optimizer.py` (255 lines)
- Intent: Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.
- Classes: none
- Top-level functions:
  - `optimize_portfolio`: Find optimal portfolio weights via mean-variance optimization.

### `risk/portfolio_risk.py` (327 lines)
- Intent: Portfolio Risk Manager — enforces sector, correlation, and exposure limits.
- Classes:
  - `RiskCheck`: Result of a portfolio risk check. Methods: none.
  - `PortfolioRiskManager`: Enforces portfolio-level risk constraints. Methods: `__init__`, `_infer_ticker_from_price_df`, `_resolve_sector`, `check_new_position`, `_check_correlations`, `_estimate_portfolio_beta`, `_estimate_portfolio_vol`, `portfolio_summary`.
- Top-level functions: none

### `risk/position_sizer.py` (290 lines)
- Intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- Classes:
  - `PositionSize`: Result of position sizing calculation. Methods: none.
  - `PositionSizer`: Multi-method position sizer with conservative blending. Methods: `__init__`, `size_position`, `_kelly`, `_vol_scaled`, `_atr_based`, `size_portfolio`.
- Top-level functions: none

### `risk/stop_loss.py` (251 lines)
- Intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- Classes:
  - `StopReason`: No class docstring. Methods: none.
  - `StopResult`: Result of stop-loss evaluation. Methods: none.
  - `StopLossManager`: Multi-strategy stop-loss manager. Methods: `__init__`, `evaluate`, `compute_initial_stop`, `compute_risk_per_share`.
- Top-level functions: none

### `risk/stress_test.py` (363 lines)
- Intent: Stress Testing Module --- scenario analysis and historical drawdown replay.
- Classes: none
- Top-level functions:
  - `_estimate_portfolio_beta`: Estimate weighted-average beta of the portfolio vs equal-weight market proxy.
  - `_compute_portfolio_vol`: Annualized portfolio volatility from historical covariance.
  - `run_stress_scenarios`: Apply stress scenarios to a portfolio and estimate impact.
  - `run_historical_drawdown_test`: Replay the worst historical drawdown episodes on the portfolio.
  - `_find_drawdown_episodes`: Identify non-overlapping drawdown episodes from a return series.

## Package `tests`
- Modules: 20
### `tests/__init__.py` (2 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes: none
- Top-level functions: none

### `tests/test_autopilot_predictor_fallback.py` (53 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `AutopilotPredictorFallbackTests`: No class docstring. Methods: `test_ensure_predictor_falls_back_when_model_import_fails`.
- Top-level functions: none

### `tests/test_cache_metadata_rehydrate.py` (91 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `CacheMetadataRehydrateTests`: No class docstring. Methods: `test_rehydrate_writes_metadata_for_daily_csv`, `test_rehydrate_only_missing_does_not_overwrite`, `test_rehydrate_force_with_overwrite_source_updates_source`.
- Top-level functions:
  - `_write_daily_csv`: No function docstring.

### `tests/test_covariance_estimator.py` (20 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `CovarianceEstimatorTests`: No class docstring. Methods: `test_single_asset_covariance_is_2d_and_positive`.
- Top-level functions: none

### `tests/test_delisting_total_return.py` (71 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `DelistingTotalReturnTests`: No class docstring. Methods: `test_target_uses_total_return_when_available`, `test_indicator_values_unaffected_by_delist_return_columns`.
- Top-level functions: none

### `tests/test_drawdown_liquidation.py` (128 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `_FakePositionSizer`: No class docstring. Methods: `size_position`.
  - `_FakeDrawdownController`: No class docstring. Methods: `__init__`, `update`, `get_summary`.
  - `_FakeStopLossManager`: No class docstring. Methods: `evaluate`.
  - `_FakePortfolioRisk`: No class docstring. Methods: `check_new_position`.
  - `_FakeRiskMetrics`: No class docstring. Methods: `compute_full_report`.
  - `DrawdownLiquidationTests`: No class docstring. Methods: `test_critical_drawdown_forces_liquidation`.
- Top-level functions: none

### `tests/test_execution_dynamic_costs.py` (43 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `ExecutionDynamicCostTests`: No class docstring. Methods: `test_dynamic_costs_increase_under_stress`.
- Top-level functions: none

### `tests/test_integration.py` (556 lines)
- Intent: End-to-end integration tests for the quant engine pipeline.
- Classes:
  - `TestFullPipelineSynthetic`: End-to-end test: data -> features -> regimes -> training -> prediction -> backtest. Methods: `synthetic_data`, `pipeline_outputs`, `test_features_shape`, `test_targets_shape`, `test_regimes_aligned`, `test_pit_no_future_in_features`, `test_pit_no_future_in_targets`, `test_training_produces_result`.
  - `TestCvGapHardBlock`: Verify that the CV gap hard block rejects overfit models. Methods: `test_cv_gap_hard_block`.
  - `TestRegime2Suppression`: Verify regime 2 gating suppresses trades. Methods: `test_regime_2_suppression`, `test_regime_0_not_suppressed`.
  - `TestCrossSectionalRanking`: Verify cross-sectional ranker produces valid output. Methods: `test_cross_sectional_rank_basic`, `test_cross_sectional_rank_multiindex`, `test_cross_sectional_rank_zscore_centered`, `test_cross_sectional_rank_signals_count`.
- Top-level functions:
  - `_generate_synthetic_ohlcv`: Generate synthetic OHLCV data for *n_stocks* over *n_days*.

### `tests/test_iv_arbitrage_builder.py` (33 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `ArbitrageFreeSVIBuilderTests`: No class docstring. Methods: `test_build_surface_has_valid_shape_and_monotone_total_variance`.
- Top-level functions: none

### `tests/test_kalshi_asof_features.py` (60 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `KalshiAsofFeatureTests`: No class docstring. Methods: `test_event_feature_panel_uses_backward_asof_join`, `test_event_feature_panel_raises_when_required_columns_missing`.
- Top-level functions: none

### `tests/test_kalshi_distribution.py` (109 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `KalshiDistributionTests`: No class docstring. Methods: `test_bin_distribution_normalizes_and_computes_moments`, `test_threshold_distribution_applies_monotone_constraint`, `test_distribution_panel_accepts_tz_aware_snapshot_times`.
- Top-level functions: none

### `tests/test_kalshi_hardening.py` (600 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `KalshiHardeningTests`: No class docstring. Methods: `test_bin_distribution_mass_normalizes_to_one`, `test_threshold_direction_semantics_change_tail_probabilities`, `test_unknown_threshold_direction_marked_quality_low`, `test_dynamic_stale_cutoff_tightens_near_event`, `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity`, `test_quality_score_behaves_sensibly_on_synthetic_cases`, `test_event_panel_supports_event_id_mapping`, `test_event_labels_first_vs_latest`, `test_walkforward_runs_and_counts_trials`, `test_walkforward_contract_metrics_are_computed`, `test_event_promotion_flow_uses_walkforward_contract_metrics`, `test_options_disagreement_features_are_joined_asof`, `test_mapping_store_asof`, `test_store_ingestion_and_health_tables`, `test_provider_materializes_daily_health_report`, `test_signer_canonical_payload_and_header_fields`.
- Top-level functions: none

### `tests/test_loader_and_predictor.py` (220 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `_FakeWRDSProvider`: No class docstring. Methods: `available`, `get_crsp_prices`, `get_crsp_prices_with_delistings`, `resolve_permno`.
  - `_UnavailableWRDSProvider`: No class docstring. Methods: `available`.
  - `LoaderAndPredictorTests`: No class docstring. Methods: `test_load_ohlcv_uses_wrds_contract_and_stable_columns`, `test_load_with_delistings_applies_delisting_return`, `test_predictor_explicit_version_does_not_silently_fallback`, `test_cache_load_reads_daily_csv_when_parquet_unavailable`, `test_cache_save_falls_back_to_csv_without_parquet_engine`, `test_trusted_wrds_cache_short_circuits_live_wrds`, `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source`, `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable`.
- Top-level functions: none

### `tests/test_panel_split.py` (50 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `PanelSplitTests`: No class docstring. Methods: `test_holdout_mask_uses_dates_not_raw_rows`, `test_date_purged_folds_do_not_overlap`.
- Top-level functions: none

### `tests/test_paper_trader_kelly.py` (120 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `PaperTraderKellyTests`: No class docstring. Methods: `test_kelly_sizing_changes_position_size_with_bounds`.
- Top-level functions:
  - `_mock_price_data`: No function docstring.
  - `_seed_state`: No function docstring.
  - `_run_cycle`: No function docstring.

### `tests/test_promotion_contract.py` (95 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `PromotionContractTests`: No class docstring. Methods: `test_contract_fails_when_advanced_requirements_fail`, `test_contract_passes_when_all_checks_pass`.
- Top-level functions:
  - `_candidate`: No function docstring.
  - `_result`: No function docstring.

### `tests/test_provider_registry.py` (24 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `ProviderRegistryTests`: No class docstring. Methods: `test_registry_lists_core_providers`, `test_registry_rejects_unknown_provider`, `test_registry_can_construct_kalshi_provider`.
- Top-level functions: none

### `tests/test_research_factors.py` (127 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `ResearchFactorTests`: No class docstring. Methods: `test_single_asset_research_features_exist`, `test_cross_asset_network_features_shape_and_bounds`, `test_cross_asset_factors_are_causally_lagged`, `test_pipeline_universe_includes_research_features`.
- Top-level functions:
  - `_make_ohlcv`: No function docstring.

### `tests/test_survivorship_pit.py` (69 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `SurvivorshipPointInTimeTests`: No class docstring. Methods: `test_filter_panel_by_point_in_time_universe`.
- Top-level functions: none

### `tests/test_validation_and_risk_extensions.py` (101 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes:
  - `ValidationAndRiskExtensionTests`: No class docstring. Methods: `test_cpcv_detects_positive_signal_quality`, `test_spa_passes_for_consistently_positive_signal_returns`, `test_portfolio_risk_rejects_high_projected_volatility`.
- Top-level functions:
  - `_make_ohlcv`: No function docstring.

## Package `utils`
- Modules: 2
### `utils/__init__.py` (1 lines)
- Intent: No module docstring; use names below as source of intent.
- Classes: none
- Top-level functions: none

### `utils/logging.py` (436 lines)
- Intent: Structured logging for the quant engine.
- Classes:
  - `StructuredFormatter`: JSON formatter for machine-parseable log output. Methods: `format`.
  - `AlertHistory`: Persistent alert history with optional webhook notifications. Methods: `__init__`, `_load`, `_save`, `record`, `record_batch`, `query`, `_notify_webhook`.
  - `MetricsEmitter`: Emit key metrics on every cycle and check alert thresholds. Methods: `__init__`, `emit_cycle_metrics`, `check_alerts`.
- Top-level functions:
  - `get_logger`: Get a structured logger for the quant engine.


