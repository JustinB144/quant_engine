# `autopilot` Package Guide

## Purpose

Autopilot layer: discovery, promotion, and paper-trading orchestration.

## Package Summary

- Modules: 8
- Classes: 12
- Top-level functions: 0
- LOC: 3,685

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---|---|---|---|
| `autopilot/__init__.py` | 23 | 0 | 0 | Autopilot layer: discovery, promotion, and paper-trading orchestration. |
| `autopilot/engine.py` | 1395 | 2 | 0 | End-to-end autopilot cycle: |
| `autopilot/meta_labeler.py` | 457 | 1 | 0 | Meta-labeling model for signal confidence prediction (Spec 04). |
| `autopilot/paper_trader.py` | 1094 | 1 | 0 | Stateful paper-trading engine for promoted strategies. |
| `autopilot/promotion_gate.py` | 328 | 2 | 0 | Promotion gate for deciding whether a discovered strategy is deployable. |
| `autopilot/registry.py` | 111 | 2 | 0 | Persistent strategy registry for promoted candidates. |
| `autopilot/strategy_allocator.py` | 197 | 2 | 0 | Regime-Aware Strategy Allocation — automatically adjust strategy parameters |
| `autopilot/strategy_discovery.py` | 80 | 2 | 0 | Strategy discovery for execution-layer parameter variants. |

## Module Details

### `autopilot/__init__.py`
- Intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- Classes: none
- Top-level functions: none

### `autopilot/engine.py`
- Intent: End-to-end autopilot cycle:
- Classes:
  - `HeuristicPredictor` (methods: `predict`)
  - `AutopilotEngine` (methods: `run_cycle`)
- Top-level functions: none

### `autopilot/meta_labeler.py`
- Intent: Meta-labeling model for signal confidence prediction (Spec 04).
- Classes:
  - `MetaLabelingModel` (methods: `build_meta_features`, `build_labels`, `train`, `predict_confidence`, `save`, `load`, `is_trained`)
- Top-level functions: none

### `autopilot/paper_trader.py`
- Intent: Stateful paper-trading engine for promoted strategies.
- Classes:
  - `PaperTrader` (methods: `run_cycle`)
- Top-level functions: none

### `autopilot/promotion_gate.py`
- Intent: Promotion gate for deciding whether a discovered strategy is deployable.
- Classes:
  - `PromotionDecision` (methods: `to_dict`)
  - `PromotionGate` (methods: `evaluate`, `evaluate_event_strategy`, `rank`)
- Top-level functions: none

### `autopilot/registry.py`
- Intent: Persistent strategy registry for promoted candidates.
- Classes:
  - `ActiveStrategy` (methods: `to_dict`)
  - `StrategyRegistry` (methods: `get_active`, `apply_promotions`)
- Top-level functions: none

### `autopilot/strategy_allocator.py`
- Intent: Regime-Aware Strategy Allocation — automatically adjust strategy parameters
- Classes:
  - `StrategyProfile` (methods: none)
  - `StrategyAllocator` (methods: `get_regime_profile`, `get_all_profiles`, `summarize`)
- Top-level functions: none

### `autopilot/strategy_discovery.py`
- Intent: Strategy discovery for execution-layer parameter variants.
- Classes:
  - `StrategyCandidate` (methods: `to_dict`)
  - `StrategyDiscovery` (methods: `generate`)
- Top-level functions: none

## Related Docs

- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md`
- `../docs/architecture/SYSTEM_CONTRACTS_AND_INVARIANTS.md`
- `../docs/reference/SOURCE_API_REFERENCE.md`
- `../docs/operations/CLI_AND_WORKFLOW_RUNBOOK.md`
