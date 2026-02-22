# Quant Engine Human System Guide

## What This Is

This is the human-use version of the system audit.
It explains what the system is trying to do, how the major components fit together, and how to use the Dash UI and operational scripts without needing to read code first.

For the full source-derived component inventory, use:
- `docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`

## System Purpose (Plain English)

`quant_engine` is a research and operations framework for:
- building predictive signals for equities,
- detecting market regimes,
- backtesting strategies with realistic risk/execution assumptions,
- managing a strategy lifecycle (discovery -> validation -> promotion -> paper trading),
- and researching event-market signals (Kalshi) in a leak-safe way.

The system is designed to avoid common research mistakes:
- survivorship bias,
- look-ahead leakage,
- overfitting via too many variants,
- fragile ticker-based joins,
- and promoting strategies just because a backtest looks good.

## Core Mental Model (How the System Works)

The system is easiest to understand as a pipeline:

1. Data
- Historical price/fundamental/options data is loaded and normalized.
- Point-in-time handling and data quality checks happen here.

2. Features
- Market and research features are engineered from the raw data.
- This includes technical, cross-asset, macro, options, and optional intraday-style features.

3. Regimes
- The market state is classified (bull/bear/mean-reverting/high-vol), with confidence.
- Correlation regime features can also be added.

4. Model
- A predictive model (with global + regime-aware behavior) is trained and versioned.
- Predictions include confidence and can be filtered/gated.

5. Backtest + Risk
- Signals are converted into trades with execution costs and risk controls.
- Performance and robustness metrics are computed.

6. Autopilot (optional)
- Execution parameter variants are discovered and evaluated.
- Only robust variants are promoted.
- Promoted strategies can be paper traded.

7. UI (Dash)
- The Dash app is the current UI to inspect health, data, models, backtests, IV surfaces, and autopilot/Kalshi workflows.

## Most Important System Rules (You Should Know These)

### 1. PERMNO-first identity
The system is built to prefer stable security identity keys (`permno`) over tickers.

Why this matters:
- Tickers can change or be reused.
- Historical joins and backtests are safer with stable identifiers.

Practical implication:
- If something looks wrong in joins or predictions, check identity handling first.

### 2. Strict time alignment (no leakage)
The system is designed to avoid using future information when building features or labels.

Why this matters:
- Leakage creates fake performance.
- This is especially critical in the Kalshi/event pipeline.

Practical implication:
- Any new feature or label logic must preserve strict as-of behavior.

### 3. Promotion gating is stricter than backtest performance
A strategy can have a decent backtest and still fail promotion.

Why this matters:
- The system checks robustness (walk-forward behavior, advanced validation, capacity constraints, etc.).

Practical implication:
- “Why wasn’t it promoted?” is often answered by the promotion gate, not by the backtest summary.

## Operational Scripts (What To Run)

These are the primary operator entry points:

- `run_dash.py`
  - Launches the current Dash UI (`dash_ui`)

- `run_train.py`
  - Trains and saves model artifacts

- `run_predict.py`
  - Generates predictions from a saved model version

- `run_backtest.py`
  - Runs backtests using predictions and price data

- `run_retrain.py`
  - Runs retrain governance / controlled retraining workflows

- `run_autopilot.py`
  - Runs strategy discovery, validation, promotion, and paper trading cycle

- `run_kalshi_event_pipeline.py`
  - Runs Kalshi event-market ingestion / feature / walk-forward / promotion workflow

- `run_wrds_daily_refresh.py`
  - Refreshes WRDS-derived caches/data pipelines

## Dash UI Guide (Current UI)

The active UI is Dash (`dash_ui`).
The old `ui/` system has been removed.

### Dashboard (`/`)
What it is for:
- Quick overview of portfolio/system state
- KPI cards and high-level status signals

Use it when:
- You want a fast top-level pulse check before deeper analysis

### System Health (`/system-health`)
What it is for:
- Health/quality checks across system dimensions
- Operational readiness and diagnostics

Use it when:
- You want to confirm the system is in a good state before training/backtesting/autopilot runs

### Data Explorer (`/data-explorer`)
What it is for:
- Inspect loaded OHLCV data and quality details
- Verify what data is actually available before modeling/backtesting

Use it when:
- Data looks suspicious
- A run fails due to missing data
- You need to inspect a specific ticker/asset series visually

### Model Lab (`/model-lab`)
What it is for:
- Feature inspection
- Regime detection visualization
- Training workflow experiments and diagnostics

Use it when:
- You are iterating on features/regimes/model settings
- You want to inspect regime probabilities and transitions visually

### Signal Desk (`/signal-desk`)
What it is for:
- Generate and inspect ranked predictions/signals
- Review prediction distributions and confidence patterns

Use it when:
- You want to see what the model is currently predicting and how confident it is

### Backtest & Risk (`/backtest-risk`)
What it is for:
- Run backtests (UI-driven parameter experiments)
- Inspect equity curve, drawdown, returns distribution, and trade/regime analytics

Use it when:
- You want a fast strategy behavior check without leaving the UI

### IV Surface (`/iv-surface`)
What it is for:
- Explore implied volatility surfaces (SVI, Heston, arbitrage-aware SVI)
- Visualize smiles and surface changes

Use it when:
- You are researching options/volatility structure or validating IV modeling behavior

### S&P Comparison (`/sp-comparison`)
What it is for:
- Compare strategy behavior vs S&P benchmark metrics/curves
- Benchmark-oriented diagnostics (alpha/beta/tracking behavior)

Use it when:
- You want to evaluate strategy behavior relative to a market benchmark, not just absolute returns

### Autopilot & Events (`/autopilot`)
What it is for:
- View strategy discovery/promotion funnel
- Inspect paper-trading outputs
- Explore Kalshi event-market visuals and walk-forward outputs

Use it when:
- You are operating the strategy lifecycle or reviewing event-market research outputs

## Subsystem Guide (What Each Area Is Meant To Do)

### `data/`
Purpose:
- Fetch, cache, and normalize data from providers (especially WRDS) with quality/provenance control.

Why it exists:
- So every downstream workflow uses the same trusted loading behavior.

### `features/`
Purpose:
- Build model-ready feature panels from market and related data.

Why it exists:
- Centralizes feature definitions and keeps train/predict/backtest consistent.

### `regime/`
Purpose:
- Detect market regime and regime confidence.

Why it exists:
- Model behavior and risk behavior should change with market conditions.

### `models/`
Purpose:
- Train, version, load, and govern predictive models.

Why it exists:
- Separates ML lifecycle concerns from data and backtest logic.

### `backtest/`
Purpose:
- Simulate strategy behavior from predictions under realistic constraints.

Why it exists:
- Converts signal quality into trading performance evidence.

### `risk/`
Purpose:
- Position sizing, risk limits, drawdown controls, portfolio risk, stress testing, attribution.

Why it exists:
- Keeps risk logic reusable and consistent across backtest/autopilot workflows.

### `autopilot/`
Purpose:
- Strategy lifecycle management: discovery, promotion, registry, paper trading.

Why it exists:
- Moves the system from manual strategy tweaking to a repeatable validation/promotion process.

### `kalshi/`
Purpose:
- Event-market data ingestion, distribution building, event features, walk-forward validation, event strategy promotion.

Why it exists:
- Event-market research has different data/timing semantics and needs its own isolated pipeline.

### `dash_ui/`
Purpose:
- Operational/research interface for humans.

Why it exists:
- Centralized visibility and control without repeatedly running scripts blind.

## Regime Detection (Human Version)

The system uses regimes to describe market behavior and adapt decisions.

Canonical regime labels used across the system:
- `0` = bull trend
- `1` = bear trend
- `2` = mean-reverting / neutral-ish
- `3` = high volatility / stressed

There are two ways regime detection can run:
- Rule-based (deterministic thresholds)
- HMM-based (probabilistic hidden-state model)

Why both exist:
- Rule-based is simple and robust as a fallback.
- HMM-based is better for latent/noisy state transitions and confidence/posteriors.

How it affects the rest of the system:
- Prediction blending/weighting
- Confidence handling
- Risk sizing and gating
- Backtest trade filtering in some conditions
- Autopilot/paper trading exposure behavior

## Autopilot (Human Version)

Autopilot does not create an entirely new predictive model by itself.
It mostly searches execution-layer variants around a predictive model.

What it does in practice:
- Ensures a baseline predictor exists (or uses a fallback predictor if necessary)
- Generates strategy candidates (thresholds, risk mode, max positions)
- Backtests candidates
- Applies promotion gates (quality and robustness rules)
- Stores promoted strategies in a registry
- Runs paper trading for active promoted strategies

How to think about it:
- It is a strategy operations layer sitting on top of the ML prediction stack.

## Kalshi Event Pipeline (Human Version)

The Kalshi subsystem converts event-market quotes into structured, time-safe event features.

High-level process:
- pull markets/contracts/quotes,
- build probability-like distribution snapshots from contract prices,
- build pre-event features for known macro events,
- run event walk-forward validation,
- apply promotion gating to event strategies.

What makes it special:
- Very strict timestamp handling (feature rows must be pre-release)
- Data-quality and quote-staleness logic is built into the feature generation path
- Promotion uses advanced validation concepts, not just raw average return

## Typical Human Workflows

### Daily/regular health workflow
1. Launch `run_dash.py`
2. Check `/system-health`
3. Check `/dashboard`
4. Use `/data-explorer` if data quality looks off
5. Use `/autopilot` to inspect promotion/paper-trading state

### Model iteration workflow
1. Inspect data in `/data-explorer`
2. Inspect features/regimes in `/model-lab`
3. Train via `run_train.py` (or use Model Lab UI tools for inspection)
4. Generate signals in `/signal-desk`
5. Backtest in `/backtest-risk` or `run_backtest.py`

### Autopilot review workflow
1. Run `run_autopilot.py`
2. Open `/autopilot`
3. Review strategy table, promotion funnel, and paper equity/positions
4. Investigate failures via promotion-gate metrics (not just returns)

### Kalshi/event research workflow
1. Run `run_kalshi_event_pipeline.py`
2. Open `/autopilot` (Kalshi sections)
3. Review probability/disagreement/walk-forward visuals
4. Confirm event feature quality and promotion outcomes

## What Changed in This Cleanup (Important)

- Legacy UI (`ui/` + `run_dashboard.py`) was removed.
- The Dash UI (`dash_ui`) is the only active UI stack.
- Loose root docs were organized into `docs/guides`, `docs/reports`, `docs/plans`, and `docs/notes`.

## Recommended Reading Order (Human)

1. `docs/guides/QUANT_ENGINE_HUMAN_SYSTEM_GUIDE.md` (this file)
2. `docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep source-based audit)
3. `docs/guides/DASH_QUICK_START.md` (UI-specific usage)
4. `docs/guides/DASH_FOUNDATION_SUMMARY.md` (UI architecture summary)
