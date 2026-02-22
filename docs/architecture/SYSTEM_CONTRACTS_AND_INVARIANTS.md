# System Contracts and Invariants

## Purpose

This document captures the most important behavior contracts across `quant_engine`.
If you change code that touches these contracts, you must assess downstream impact.

## Identity Contracts

### PERMNO-first core identity
- Canonical identity is `permno` where available.
- `ticker` is tolerated for legacy compatibility and UI display, not preferred for core joins.

Why:
- Historical ticker instability causes silent errors in joins/backtests.

Impacted subsystems:
- `data.loader`
- `data.wrds_provider`
- `backtest.engine`
- `autopilot.engine`
- `autopilot.paper_trader`

## Time / Leakage Contracts

### Strict as-of joins
- Feature joins must be backward (no forward peeking).
- Event feature rows must satisfy `asof_ts < release_ts`.

Why:
- Look-ahead bias invalidates model/backtest/event results.

Explicit guard locations (source):
- `kalshi.events.asof_join(...)`
- `kalshi.events._ensure_asof_before_release(...)`
- `kalshi.walkforward` purge/embargo logic
- `models.trainer` panel-safe split / purged CV logic

## Data Quality / Provenance Contracts

### Trusted-source preference
- Core data loading prefers trusted cache/WRDS data paths before weaker fallbacks.

Why:
- Quality and provenance must be explicit and reproducible.

### Quality dimensions are first-class
- Low quality should be represented as scores/flags/NaN outputs where appropriate, not hidden.

Why:
- Downstream models and operators need to know when data quality is poor.

Examples:
- `data.quality`
- `kalshi.quality`
- `kalshi.distribution` (`quality_low`, cleaning metrics, confidence flags)

## Regime Contracts

### Canonical regime labels
- `0`: bull trend
- `1`: bear trend
- `2`: mean reverting / default
- `3`: high volatility

Why:
- These labels are used across training, prediction, backtest, autopilot, and UI.

### Regime outputs include confidence
- Regime is not only a label; confidence/posteriors are part of the contract.

Why:
- Downstream suppression/blending/sizing logic uses confidence-aware behavior.

## Model / Artifact Contracts

### Versioning and champion resolution
- `models.predictor` and training outputs rely on versioning metadata and registries.
- Explicit version requests should not silently redirect to a different model version.

Why:
- Silent fallback creates impossible-to-debug mismatches between expected and actual behavior.

## Backtest / Execution Contracts

### Execution realism is part of the system design
- Transaction costs, slippage/impact, and realistic fills are not optional aesthetics.
- Backtests are intended to be execution-aware.

Why:
- Strategy quality is judged by simulated tradable performance, not raw signal correlation alone.

### Result schema stability matters
- Backtest metrics/results feed autopilot promotion checks and UI pages.

Why:
- Breaking field names/types can silently break ranking, promotion, and dashboards.

## Promotion Contracts

### Promotion != backtest pass
- Promotion requires passing `autopilot.promotion_gate` checks.
- Advanced validation metrics (DSR/PBO/capacity/walk-forward robustness) are part of the deployability decision.

Why:
- This is the system’s anti-overfitting and operational-risk filter.

### Event strategies use shared standards plus event-specific constraints
- Kalshi event strategies route through shared promotion logic and add event-specific checks.

Why:
- Maintains consistent quality bar across strategy types.

## Kalshi Event Pipeline Contracts

### Storage schema compatibility
- `kalshi.storage` is a stable event-time persistence layer.
- Prefer additive schema changes.

Why:
- Multiple provider/feature/evaluation layers depend on table and column stability.

### Distribution reconstruction exposes uncertainty and repairs
- Threshold-direction confidence, monotonic repairs, stale quote filtering, and quality flags are intentional outputs.

Why:
- Event features should reflect uncertainty in the underlying market representation.

## UI Contracts

### Active UI stack is Dash
- Use `dash_ui/*` and `run_dash.py`.
- Legacy `ui/` stack has been removed.

### Page-level separation is intentional
- Pages represent distinct operational questions/workflows.
- Prefer page-local modifications when possible.

Why:
- Limits blast radius and improves maintainability for both humans and LLMs.

## Testing as Contract Enforcement

Treat tests as executable behavioral specs for:
- leakage prevention
- identity correctness / delisting handling
- execution-cost realism
- autopilot fallback behavior
- Kalshi hardening (signing, stale quotes, threshold semantics, bin validity, walk-forward purge)

Changing behavior in these areas should be deliberate and documented.
