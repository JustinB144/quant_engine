# `autopilot` Package Guide

## Purpose

Autopilot layer: discovery, promotion, and paper-trading orchestration.

## Package Summary

- Modules: 6
- Classes: 9
- Top-level functions: 0
- LOC: 2,009

## How This Package Fits Into The System

- Consumes prediction, backtest, validation, and risk outputs to manage strategy lifecycle decisions.
- Persists state to `results/autopilot/*` (`strategy_registry.json`, `paper_state.json`, `latest_cycle.json`).
- Exposed to the web app through `api/routers/autopilot.py` and used by the React `/autopilot` page.

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `autopilot/__init__.py` | 20 | 0 | 0 | Autopilot layer: discovery, promotion, and paper-trading orchestration. |
| `autopilot/engine.py` | 990 | 2 | 0 | End-to-end autopilot cycle: |
| `autopilot/paper_trader.py` | 529 | 1 | 0 | Stateful paper-trading engine for promoted strategies. |
| `autopilot/promotion_gate.py` | 281 | 2 | 0 | Promotion gate for deciding whether a discovered strategy is deployable. |
| `autopilot/registry.py` | 110 | 2 | 0 | Persistent strategy registry for promoted candidates. |
| `autopilot/strategy_discovery.py` | 79 | 2 | 0 | Strategy discovery for execution-layer parameter variants. |

## Module Details

### `autopilot/__init__.py`
- Intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- Classes: none
- Top-level functions: none

### `autopilot/engine.py`
- Intent: End-to-end autopilot cycle:
- Classes:
  - `HeuristicPredictor`: Lightweight fallback predictor used when sklearn-backed model artifacts
    - Methods: `predict`
  - `AutopilotEngine`: Coordinates discovery, promotion, and paper execution.
    - Methods: `run_cycle`
- Top-level functions: none

### `autopilot/paper_trader.py`
- Intent: Stateful paper-trading engine for promoted strategies.
- Classes:
  - `PaperTrader`: Executes paper entries/exits from promoted strategy definitions.
    - Methods: `run_cycle`
- Top-level functions: none

### `autopilot/promotion_gate.py`
- Intent: Promotion gate for deciding whether a discovered strategy is deployable.
- Classes:
  - `PromotionDecision`: Serializable promotion-gate decision for a single strategy candidate evaluation.
    - Methods: `to_dict`
  - `PromotionGate`: Applies hard risk/quality constraints before a strategy can be paper-deployed.
    - Methods: `evaluate`, `evaluate_event_strategy`, `rank`
- Top-level functions: none

### `autopilot/registry.py`
- Intent: Persistent strategy registry for promoted candidates.
- Classes:
  - `ActiveStrategy`: Persisted record for a currently active promoted strategy.
    - Methods: `to_dict`
  - `StrategyRegistry`: Maintains promoted strategy state and historical promotion decisions.
    - Methods: `get_active`, `apply_promotions`
- Top-level functions: none

### `autopilot/strategy_discovery.py`
- Intent: Strategy discovery for execution-layer parameter variants.
- Classes:
  - `StrategyCandidate`: Execution-parameter variant generated for backtest and promotion evaluation.
    - Methods: `to_dict`
  - `StrategyDiscovery`: Generates a deterministic candidate grid for backtest validation.
    - Methods: `generate`
- Top-level functions: none

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (current runtime architecture)
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md` (cross-module constraints)
- `../docs/reference/SOURCE_API_REFERENCE.md` (source-derived Python module inventory)
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md` (entrypoints and workflows)
