# DEPENDENCY MATRIX
Generated: 2026-02-27

## Module-Level Adjacency Matrix

Each cell shows the number of cross-module import edges from the row module to the column module.

| Source ↓ / Target → | config | data | features | indicators | regime | models | backtest | risk | evaluation | validation | autopilot | kalshi | api | utils |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **config** | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **data** | 10 | . | . | . | . | . | . | . | . | 1 | . | 1 | . | . |
| **features** | 8 | 3 | . | 6 | 1 | . | . | . | . | . | . | . | . | . |
| **indicators** | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **regime** | 11 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **models** | 7 | . | 1 | . | . | . | . | . | . | 1 | . | . | . | . |
| **backtest** | 5 | . | . | . | 3 | . | . | 5 | . | 1 | . | . | . | . |
| **risk** | 8 | . | . | . | 1 | . | . | . | . | . | . | . | . | . |
| **evaluation** | 7 | . | . | . | . | 1 | 2 | . | . | . | . | . | . | . |
| **validation** | 2 | 1 | . | . | . | . | . | . | . | . | . | . | . | . |
| **autopilot** | 8 | 2 | 1 | . | 2 | 5 | 8 | 7 | . | . | . | . | 6 | . |
| **kalshi** | 1 | . | 1 | . | . | . | 2 | . | . | . | 3 | . | . | . |
| **api** | 82 | 7 | 2 | . | 4 | 13 | 3 | 1 | . | . | 1 | 2 | . | . |
| **utils** | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| **entry_points** | 9 | 12 | 4 | . | 4 | 9 | 3 | . | . | . | 1 | 5 | 2 | . |
| **scripts** | 2 | 1 | 1 | . | 2 | . | . | . | . | . | . | . | . | . |

## Fan-Out (Outgoing Dependencies)

Number of cross-module import edges originating from each module.

| Module | Fan-Out (edges) | Target Modules |
|---|---:|---|
| config | 0 | (none) |
| data | 12 | config, validation, kalshi |
| features | 18 | config, data, indicators, regime |
| indicators | 0 | (none) |
| regime | 11 | config |
| models | 9 | config, features, validation |
| backtest | 14 | config, regime, risk, validation |
| risk | 9 | config, regime |
| evaluation | 10 | config, models, backtest |
| validation | 3 | config, data |
| autopilot | 39 | config, data, features, regime, models, backtest, risk, api |
| kalshi | 7 | config, features, backtest, autopilot |
| api | 115 | config, data, features, regime, models, backtest, risk, autopilot, kalshi |
| utils | 1 | config |
| entry_points | 49 | config, data, features, regime, models, backtest, autopilot, kalshi, api |
| scripts | 6 | config, data, features, regime |

## Fan-In (Incoming Dependencies)

Number of cross-module import edges targeting each module.

| Module | Fan-In (edges) | Source Modules |
|---|---:|---|
| config | 161 | data, features, regime, models, backtest, risk, evaluation, validation, autopilot, kalshi, api, utils, entry_points, scripts |
| data | 26 | features, validation, autopilot, api, entry_points, scripts |
| features | 10 | models, autopilot, kalshi, api, entry_points, scripts |
| indicators | 6 | features |
| regime | 17 | features, backtest, risk, autopilot, api, entry_points, scripts |
| models | 28 | evaluation, autopilot, api, entry_points |
| backtest | 18 | evaluation, autopilot, kalshi, api, entry_points |
| risk | 13 | backtest, autopilot, api |
| evaluation | 0 | (none) |
| validation | 3 | data, models, backtest |
| autopilot | 5 | kalshi, api, entry_points |
| kalshi | 8 | data, api, entry_points |
| api | 8 | autopilot, entry_points |
| utils | 0 | (none) |

## Hub Modules (5+ cross-module connections)

- **config**: 14 distinct module connections (imports from 0 modules, imported by 14 modules)
- **data**: 8 distinct module connections (imports from 3 modules, imported by 6 modules)
- **features**: 10 distinct module connections (imports from 4 modules, imported by 6 modules)
- **regime**: 8 distinct module connections (imports from 1 modules, imported by 7 modules)
- **models**: 7 distinct module connections (imports from 3 modules, imported by 4 modules)
- **backtest**: 9 distinct module connections (imports from 4 modules, imported by 5 modules)
- **risk**: 5 distinct module connections (imports from 2 modules, imported by 3 modules)
- **autopilot**: 10 distinct module connections (imports from 8 modules, imported by 3 modules)
- **kalshi**: 7 distinct module connections (imports from 4 modules, imported by 3 modules)
- **api**: 10 distinct module connections (imports from 9 modules, imported by 2 modules)
- **entry_points**: 9 distinct module connections (imports from 9 modules, imported by 0 modules)

## Isolated Modules (zero cross-module imports)

(None — all modules participate in cross-module imports)

## Import Type Breakdown

- **Top-level**: 119 edges (loaded at import time)
- **Lazy**: 91 edges (loaded inside function bodies)
- **Conditional**: 98 edges (inside try/except or if-guards)

## Architectural Notes

- `autopilot/engine.py:1868` → `api`: Circular: autopilot imports from api (api serves autopilot)
- `autopilot/engine.py:1911` → `api`: Circular: autopilot imports from api (api serves autopilot)
- `autopilot/paper_trader.py:173` → `api`: Circular: autopilot imports from api (api serves autopilot)
- `autopilot/paper_trader.py:189` → `api`: Circular: autopilot imports from api (api serves autopilot)
- `autopilot/paper_trader.py:532` → `api`: Circular: autopilot imports from api (api serves autopilot)
- `autopilot/paper_trader.py:211` → `api`: Circular: autopilot imports from api (api serves autopilot)
- `data/provider_registry.py:23` → `kalshi`: Lazy import inside factory function — optional coupling
