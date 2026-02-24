# System Contracts and Invariants

This document records the highest-value contracts enforced (or assumed) across the current source tree. When behavior changes in these areas, update both code and docs deliberately.

## Identity Contracts

### PERMNO-first equity identity
- Canonical identity for core equity workflows is `permno` when available.
- `ticker` is a display/access convenience, not the primary join key for historical correctness.
- Guardrails appear in `data/loader.py`, `backtest/engine.py`, and `autopilot/engine.py` (PERMNO assertions / validation helpers).

## Time / Leakage Contracts

### As-of joins must be backward-looking
- Feature and event joins must not use future information.
- Kalshi event features enforce strict as-of semantics in `kalshi/events.py` and are tested in `tests/test_kalshi_asof_features.py` and `kalshi/tests/test_no_leakage.py`.

### Validation must preserve time ordering
- Purged/embargoed splits and walk-forward logic are part of the correctness contract, not optional extras.
- Source anchors: `models/trainer.py`, `backtest/validation.py`, `kalshi/walkforward.py`.

## Data Quality / Provenance Contracts

### Provenance and cache trust are explicit
- Cache source trust/freshness is modeled in `data/local_cache.py` + `data/loader.py` and surfaced by `/api/data/status`.
- Low-quality or fallback conditions should be represented in outputs/metadata, not silently hidden.

### Configuration statuses are documented in source
- `config.py` uses inline `STATUS:` annotations (`ACTIVE`, `PLACEHOLDER`, etc.) and `/api/config/status` reflects effective feature status, including inactive constraints when prerequisites (e.g., `GICS_SECTORS`) are missing.

## Regime Contracts

### Canonical regime labels and names
- Regime IDs are defined in `config.py` (`REGIME_NAMES`): `0 trending_bull`, `1 trending_bear`, `2 mean_reverting`, `3 high_volatility`.
- These IDs flow across training, prediction, backtests, autopilot, and UI displays.

### Regime probabilities/confidence are part of the interface
- Regime outputs are not just labels; confidence/probability columns are used for suppression, blending, and UI diagnostics.

## Model / Artifact Contracts

### Versioned artifact directories are the runtime truth
- Training writes artifacts to `trained_models/` and version metadata in registries (`models/versioning.py`, `models/governance.py`).
- Prediction/backtest services often read persisted artifacts/results rather than recomputing in request handlers.

### Explicit version requests should not silently change
- Tests guard against silent fallback behavior (`tests/test_loader_and_predictor.py`).

## Backtest / Execution Contracts

### Execution realism is part of system correctness
- Costs, fills, participation limits, and dynamic execution assumptions are built into the backtest path (`backtest/engine.py`, `backtest/execution.py`).
- Promotion and UI analytics consume these outputs, so simplifying away execution realism changes the system contract.

### Result file schemas are shared interfaces
- `results/backtest_{horizon}d_summary.json`, `results/backtest_{horizon}d_trades.csv`, and `results/predictions_{horizon}d.csv` are read by API services and the frontend via API endpoints.
- Schema changes must be coordinated across engine writers, service readers, and frontend types/components.

## Promotion / Autopilot Contracts

### Promotion requires contract checks, not just performance
- `autopilot/promotion_gate.py` enforces hard thresholds and validation requirements (DSR/PBO/CPCV/etc. depending config flags).
- A good backtest is insufficient without passing promotion constraints.

### Autopilot state files are stable shared artifacts
- `results/autopilot/strategy_registry.json`, `paper_state.json`, and `latest_cycle.json` are consumed by API service adapters and the frontend autopilot page.

## Kalshi Contracts

### Storage schema compatibility
- `kalshi/storage.py` defines a stable `EventTimeStore` schema used by ingestion, distribution, event features, and walk-forward tooling.
- Prefer additive schema changes; update `docs/reference/KALSHI_STORAGE_SCHEMA_REFERENCE.md` whenever DDL changes.

### Distribution quality/repair outputs are intentional
- Threshold direction inference, monotonic repairs, quote staleness filtering, and quality flags are part of the Kalshi data contract (`kalshi/distribution.py`, `kalshi/quality.py`).

## API Contracts

### Shared response envelope
- All API routes are expected to return `ApiResponse` envelopes (`api/schemas/envelope.py`) with `ok`, `data`, `error`, and `meta`.
- Frontend `api/client.ts` assumes this envelope for all requests.

### Background job lifecycle
- Long-running compute routes submit jobs via `api/jobs/*` and stream events via SSE.
- Job persistence status values are canonical in `api/jobs/models.py` (`queued`, `running`, `succeeded`, `failed`, `cancelled`).

## Frontend Contracts

### Active UI stack is React/Vite (`frontend/`) 
- Current UI code lives in `frontend/src/*` and is served either by Vite (dev) or `run_server.py --static` (production-like mode).
- `dash_ui/` and `run_dash.py` are removed and must not be treated as current runtime components.

### Route/page separation is intentional
- Route composition in `frontend/src/App.tsx` and navigation in `frontend/src/components/layout/Sidebar.tsx` define the operational page contract.
- Prefer page-local changes plus shared hook/component abstractions over cross-page duplication.

### Known frontend/backend contract drift (current source)
- Frontend job type definitions (`frontend/src/types/jobs.ts`) currently use `pending/completed` and `message`, while backend job models emit `queued/succeeded` and `progress_message`.
- Treat backend API models as canonical until the drift is resolved in code.

## Tests As Contract Enforcement

Use `docs/reference/TEST_SPEC_MAP.md` plus the test files themselves as executable specs for: identity, leakage, survivorship, execution realism, autopilot fallback behavior, promotion contracts, Kalshi hardening, and API envelopes/routers.
