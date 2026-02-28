# Portfolio Layer Guide — Regime-Conditioned Risk Management

> Spec 07: Regime-Conditioned Portfolio Risk Manager

## Architecture Overview

The portfolio risk layer enforces pre-trade constraints that adapt to
market regime. It comprises four interconnected components:

```
PortfolioRiskManager
├── ConstraintMultiplier       ← regime-aware multipliers (smooth transitions)
├── FactorExposureManager      ← beta/size/value/momentum/volatility bounds
├── CovarianceEstimator        ← Ledoit-Wolf / EWMA covariance
└── UniverseConfig             ← centralized sector/liquidity/borrowability YAML
```

**Data flow for `check_new_position()`:**

1. Compute effective constraint limits (base × regime multiplier × correlation stress).
2. Run single-name, gross exposure, sector, correlation, beta, and volatility checks.
3. Compute factor exposures and check against regime-conditioned bounds.
4. If any constraint utilization > 70%, compute sizing backoff recommendation.
5. Return `RiskCheck` with pass/fail, violations, metrics, utilization, and recommended weights.

---

## Configuration Reference

### `config_data/universe.yaml`

The authoritative source for universe metadata. Required sections:

| Section | Description |
|---------|-------------|
| `sectors` | `{sector_name: [ticker, ...]}` — ticker-to-sector mapping |
| `constraint_base` | Base constraint thresholds (sector_cap, correlation_limit, gross_exposure, single_name_cap, annualized_turnover_max) |
| `stress_multipliers` | `{normal: {...}, stress: {...}}` — regime multiplier sets |

Optional sections:

| Section | Description |
|---------|-------------|
| `liquidity_tiers` | `{tier_name: {market_cap_min, dollar_volume_min}}` |
| `borrowability` | `{hard_to_borrow: [...], restricted: [...]}` |
| `factor_limits` | `{factor: {normal: [lo, hi], stress: [lo, hi]}}` — null = monitored only |
| `backoff_policy` | `{mode, thresholds: [...], backoff_factors: [...]}` |

### Environment Variable Overrides

| Variable | Effect |
|----------|--------|
| `CONSTRAINT_STRESS_SECTOR_CAP` | Override stress sector cap multiplier |
| `CONSTRAINT_STRESS_CORRELATION_LIMIT` | Override stress correlation multiplier |
| `FACTOR_BETA_MIN` | Override minimum beta bound (both regimes) |
| `FACTOR_BETA_MAX` | Override maximum beta bound (both regimes) |
| `CONSTRAINT_MULTIPLIER_SMOOTHING_ALPHA` | Smoothing factor for regime transitions (default 0.3) |

---

## API Reference

### `PortfolioRiskManager`

```python
from quant_engine.risk import PortfolioRiskManager

rm = PortfolioRiskManager(
    max_sector_pct=0.40,         # base sector cap
    max_corr_between=0.85,       # base correlation limit
    max_gross_exposure=1.0,      # base gross exposure limit
    max_single_name_pct=0.10,    # base single-name cap
    max_beta_exposure=1.5,       # portfolio beta limit
    max_portfolio_vol=0.30,      # annualized vol cap
    correlation_lookback=60,     # bars for correlation calc
    covariance_method="ledoit_wolf",
    universe_config=None,        # UniverseConfig instance (auto-loads if available)
)
```

#### `check_new_position()`

Pre-trade risk check for a proposed new position.

```python
risk_check = rm.check_new_position(
    ticker="AAPL",
    position_size=0.05,
    current_positions={"MSFT": 0.10, "JNJ": 0.05},
    price_data=price_data,         # {ticker: OHLCV DataFrame}
    benchmark_data=spy_ohlcv,      # optional, for beta calc
    regime=0,                      # optional, 0-3
    regime_labels=regime_series,   # optional, for regime-conditional covariance
)

# Returns RiskCheck:
risk_check.passed                  # bool — all constraints satisfied
risk_check.violations              # List[str] — human-readable violation messages
risk_check.metrics                 # Dict — computed metrics (regime, beta, vol, factor_exposures, etc.)
risk_check.constraint_utilization  # Dict[str, float] — utilization ratios (0-1+)
risk_check.recommended_weights     # Optional[np.ndarray] — backoff-adjusted weights
```

#### `check_factor_exposures()`

Standalone factor exposure check.

```python
result = rm.check_factor_exposures(
    positions={"AAPL": 0.25, "MSFT": 0.25, "JNJ": 0.25, "JPM": 0.25},
    price_data=price_data,
    benchmark_data=spy_ohlcv,  # optional
    regime=0,                  # optional
)

result["exposures"]   # Dict[str, float] — {beta, size, value, momentum, volatility}
result["passed"]      # bool
result["violations"]  # Dict[str, str] — factor -> violation message
```

#### `compute_constraint_utilization()`

```python
util = rm.compute_constraint_utilization(
    positions={"AAPL": 0.30, "MSFT": 0.10},
    price_data=price_data,
    regime=0,
)
# Returns: {"gross_exposure": 0.40, "single_name": 3.0, "sector_cap": 0.75}
```

#### `portfolio_summary()`

```python
summary = rm.portfolio_summary(positions, price_data)
# Returns: {n_positions, gross_exposure, sector_breakdown, largest_position,
#           avg_pairwise_corr, max_pairwise_corr, corr_stress_multiplier}
```

### `ConstraintMultiplier`

```python
from quant_engine.risk import ConstraintMultiplier

cm = ConstraintMultiplier(universe_config=config)

# Raw multipliers (step function)
mults = cm.get_multipliers(regime=3)
# {"sector_cap": 0.6, "correlation_limit": 0.7, "gross_exposure": 0.8, "turnover": 0.5}

# Smoothed multipliers (exponential transition)
mults = cm.get_multipliers_smoothed(regime=3, alpha=0.3)
# Gradual transition over ~2 trading days

cm.reset()  # Reset smoothing state (e.g., start of new backtest)
```

**Regime mapping:**

| Regime | Name | Type |
|--------|------|------|
| 0 | trending_bull | Normal |
| 1 | trending_bear | Normal |
| 2 | mean_reverting | Stress |
| 3 | high_volatility | Stress |

### `FactorExposureManager`

```python
from quant_engine.risk import FactorExposureManager

fem = FactorExposureManager(universe_config=config, lookback=252, beta_lookback=60)

exposures = fem.compute_exposures(
    weights={"AAPL": 0.25, "MSFT": 0.25, "JNJ": 0.25, "JPM": 0.25},
    price_data=price_data,
    benchmark_data=spy_ohlcv,
)
# {"beta": 1.02, "size": 0.15, "value": -0.08, "momentum": 0.23, "volatility": 0.97}

passed, violations = fem.check_factor_bounds(exposures, regime=3)
```

**Default factor bounds:**

| Factor | Normal | Stress | Constrained? |
|--------|--------|--------|-------------|
| beta | [0.8, 1.2] | [0.9, 1.1] | Yes |
| volatility | [0.8, 1.2] | [0.5, 1.0] | Yes |
| size | null | null | Monitored only |
| value | null | null | Monitored only |
| momentum | null | null | Monitored only |

### `UniverseConfig`

```python
from quant_engine.risk import UniverseConfig

config = UniverseConfig("config_data/universe.yaml")

config.get_sector("AAPL")              # "tech"
config.get_sector_constituents("tech") # ["AAPL", "MSFT", ...]
config.get_all_sectors()               # ["tech", "healthcare", ...]
config.get_liquidity_tier(300e9, 2e8)  # "Mega"
config.is_hard_to_borrow("TSLA")       # True
config.is_restricted("ORCL")           # True
config.constraint_base                 # {"sector_cap": 0.40, ...}
config.stress_multipliers              # {"normal": {...}, "stress": {...}}
config.factor_limits                   # {"beta": {"normal": [0.8, 1.2], ...}}
config.backoff_policy                  # {"mode": "continuous", ...}
config.get_factor_bounds("beta", is_stress=True)  # (0.9, 1.1)
```

### Constraint Replay

```python
from quant_engine.risk import replay_with_stress_constraints, compute_robustness_score

result_df = replay_with_stress_constraints(
    portfolio_history=[
        {"date": "2023-06-01", "positions": {"AAPL": 0.30, "MSFT": 0.20}},
    ],
    price_data=price_data,
    universe_config=config,
)

score = compute_robustness_score(result_df)
# {"overall_score": 0.85, "per_scenario": {...}, "worst_scenario": "2008_crisis",
#  "avg_max_utilization": 0.72}
```

---

## Sizing Backoff Policy

When constraint utilization exceeds thresholds, position sizes are
continuously scaled down rather than gated binary:

| Utilization | Backoff Factor | Effect |
|-------------|---------------|--------|
| < 70% | 1.0 | No reduction |
| 70–80% | 0.9 | 10% reduction |
| 80–90% | 0.7 | 30% reduction |
| 90–95% | 0.5 | 50% reduction |
| > 95% | 0.25 | 75% reduction |

When multiple constraints trigger backoff, factors are multiplied:

```python
# sector_cap at 92% -> 0.5, gross at 85% -> 0.7
# Combined backoff = 0.5 * 0.7 = 0.35
```

---

## SPEC-P03: Correlation-Based Stress Tightening

In addition to regime-based multipliers, constraints tighten when average
pairwise correlation spikes — this fires faster than regime detection:

| Avg Pairwise |Corr| | Multiplier |
|---|---|
| > 0.6 | 0.85 (15% tighter) |
| > 0.7 | 0.70 (30% tighter) |
| > 0.8 | 0.50 (50% tighter) |

This multiplier stacks with the regime multiplier.

---

## Smooth Constraint Transitions

Constraint multipliers use exponential smoothing to avoid abrupt
position liquidations on regime change:

```
smoothed = alpha * target + (1 - alpha) * previous
```

Default alpha = 0.3, giving a half-life of ~2 trading days.

On regime change day 1: 30% new, 70% old.
Day 2: ~51% new. Day 3: ~65% new. Convergence after ~7 days.

Override via `CONSTRAINT_MULTIPLIER_SMOOTHING_ALPHA=1.0` for immediate
transitions (reverts to binary gating behavior).
