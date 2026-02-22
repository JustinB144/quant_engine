# `autopilot` Package Guide

## Purpose

Strategy lifecycle orchestration: candidate discovery, promotion gating, registry persistence, and paper trading.

## Package Summary

- Modules: 6
- Classes: 9
- Top-level functions: 0
- LOC: 1,770

## How This Package Fits Into the System

- Consumes `data`, `features`, `regime`, `models.predictor`, `backtest`, `risk` outputs
- Persists strategy registry and paper-trading state
- Feeds Dash `/autopilot` UI page

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
|---|---:|---:|---:|---|
| `autopilot/__init__.py` | 20 | 0 | 0 | Autopilot layer: discovery, promotion, and paper-trading orchestration. |
| `autopilot/engine.py` | 910 | 2 | 0 | End-to-end autopilot cycle: |
| `autopilot/paper_trader.py` | 432 | 1 | 0 | Stateful paper-trading engine for promoted strategies. |
| `autopilot/promotion_gate.py` | 230 | 2 | 0 | Promotion gate for deciding whether a discovered strategy is deployable. |
| `autopilot/registry.py` | 103 | 2 | 0 | Persistent strategy registry for promoted candidates. |
| `autopilot/strategy_discovery.py` | 75 | 2 | 0 | Strategy discovery for execution-layer parameter variants. |

## Module Details

### `autopilot/__init__.py`
- Intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- Classes: none
- Top-level functions: none

### `autopilot/engine.py`
- Intent: End-to-end autopilot cycle:
- Classes:
  - `HeuristicPredictor`: Lightweight fallback predictor used when sklearn-backed model artifacts
    - Methods: `__init__`, `_rolling_zscore`, `predict`
  - `AutopilotEngine`: Coordinates discovery, promotion, and paper execution.
    - Methods: `__init__`, `_log`, `_is_permno_key`, `_assert_permno_price_data`, `_assert_permno_prediction_panel`, `_assert_permno_latest_predictions`, `_load_data`, `_build_regimes`, `_train_baseline`, `_ensure_predictor`, `_predict_universe`, `_walk_forward_predictions`, `_evaluate_candidates`, `_compute_optimizer_weights`, `run_cycle`
- Top-level functions: none

### `autopilot/paper_trader.py`
- Intent: Stateful paper-trading engine for promoted strategies.
- Classes:
  - `PaperTrader`: Executes paper entries/exits from promoted strategy definitions.
    - Methods: `__init__`, `_load_state`, `_save_state`, `_resolve_as_of`, `_latest_predictions_by_id`, `_latest_predictions_by_ticker`, `_current_price`, `_position_id`, `_mark_to_market`, `_trade_return`, `_historical_trade_stats`, `_market_risk_stats`, `_position_size_pct`, `run_cycle`
- Top-level functions: none

### `autopilot/promotion_gate.py`
- Intent: Promotion gate for deciding whether a discovered strategy is deployable.
- Classes:
  - `PromotionDecision`: No class docstring.
    - Methods: `to_dict`
  - `PromotionGate`: Applies hard risk/quality constraints before a strategy can be paper-deployed.
    - Methods: `__init__`, `evaluate`, `evaluate_event_strategy`, `rank`
- Top-level functions: none

### `autopilot/registry.py`
- Intent: Persistent strategy registry for promoted candidates.
- Classes:
  - `ActiveStrategy`: No class docstring.
    - Methods: `to_dict`
  - `StrategyRegistry`: Maintains promoted strategy state and historical promotion decisions.
    - Methods: `__init__`, `_load`, `_save`, `get_active`, `apply_promotions`
- Top-level functions: none

### `autopilot/strategy_discovery.py`
- Intent: Strategy discovery for execution-layer parameter variants.
- Classes:
  - `StrategyCandidate`: No class docstring.
    - Methods: `to_dict`
  - `StrategyDiscovery`: Generates a deterministic candidate grid for backtest validation.
    - Methods: `__init__`, `generate`
- Top-level functions: none



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
