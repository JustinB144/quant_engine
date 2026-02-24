# Quant Engine — Spec-Driven Development Index

> **Total specs:** 17
> **Total estimated effort:** ~145 hours
> **Total tasks:** 103
> **Last updated:** 2026-02-23

---

## Implementation Priority

### Critical (fix before any new trading)

| # | Spec | Hours | Tasks | Focus |
|---|------|-------|-------|-------|
| 009 | [Kelly Position Sizing Overhaul](009_kelly_position_sizing_overhaul.md) | 12h | 8 | Kelly formula bug, drawdown governor, Bayesian per-regime |
| 012 | [Feature Engineering Fixes](012_feature_engineering_fixes.md) | 10h | 7 | TSMom lookahead bias, IV shock future data, feature tagging |
| 011 | [Backtest Execution Realism](011_backtest_execution_realism.md) | 10h | 7 | Entry timing, AC risk_aversion, exit volume, validation enforcement |
| 008 | [Ensemble & Conceptual Fixes](008_ensemble_and_conceptual_fixes.md) | 6h | 5 | Phantom 3rd vote, frontend defaults, Heston tab, duration smoothing |

### High (fix before trusting results)

| # | Spec | Hours | Tasks | Focus |
|---|------|-------|-------|-------|
| 010 | [Health Check Rewrites](010_health_check_rewrites.md) | 14h | 9 | All 15 checks rewritten, UNAVAILABLE status, methodology docs |
| 013 | [Model Training Pipeline](013_model_training_pipeline_fixes.md) | 8h | 6 | Per-fold feature selection, calibration split, regime min samples |
| 015 | [Paper Trader Integration](015_paper_trader_integration.md) | 8h | 6 | Wire drawdown, stops, risk manager, regime tracking, equity curve |
| 007 | [UI Errors & Log Fixes](007_ui_errors_warnings_log_fixes.md) | 8h | 6 | NaN guards, structured logging, config validation, zero-error goal |

### Medium (improve performance)

| # | Spec | Hours | Tasks | Focus |
|---|------|-------|-------|-------|
| 016 | [Risk System Improvements](016_risk_system_improvements.md) | 12h | 8 | Covariance, VaR, stress testing, attribution, spread-aware stops |
| 014 | [A/B Testing Framework](014_ab_testing_framework.md) | 10h | 7 | Block bootstrap, ticker assignment, power analysis, early stopping |
| 017 | [System-Level Innovation](017_system_level_innovation.md) | 20h | 10 | Shift detection, conformal prediction, regime allocation, diagnostics |
| 006 | [Config Cleanup](006_config_cleanup_dead_flags.md) | 4h | 4 | Dead flags, GICS_SECTORS, status annotations |

### Lower (UI and UX)

| # | Spec | Hours | Tasks | Focus |
|---|------|-------|-------|-------|
| 004 | [TradingView Charting](004_tradingview_charting.md) | 14h | 7 | Timeframe bars endpoint, indicator API, enhanced candlestick chart |
| 003 | [Regime UI & Data Fix](003_regime_ui_and_data_fix.md) | 6h | 5 | Regime API endpoint, history timeline, frontend regime tab |
| 005 | [Data Loading Diagnostics](005_data_loading_diagnostics.md) | 5h | 4 | DATA_DIR fix, /api/data/status, per-ticker cache health |
| 002 | [Health System Transparency](002_health_system_transparency.md) | 10h | 7 | HealthCheckResult dataclass, methodology panels, history |
| 001 | [Statistical Jump Model](001_statistical_jump_model.md) | 12h | 8 | Replace HMM with jumpmodels PyPI package |

---

## Cross-References

### Files modified by multiple specs

| File | Specs |
|------|-------|
| `risk/position_sizer.py` | 009, 015, 017 |
| `autopilot/paper_trader.py` | 009, 014, 015 |
| `api/services/health_service.py` | 002, 007, 010 |
| `regime/detector.py` | 003, 007, 008 |
| `backtest/engine.py` | 011, 016 |
| `backtest/advanced_validation.py` | 011 |
| `models/trainer.py` | 013, 017 |
| `models/predictor.py` | 013, 017 |
| `config.py` | 006, 007, 011, 016, 017 |
| `api/routers/data_explorer.py` | 004, 005 |
| `risk/drawdown.py` | 009, 015, 016 |
| `risk/covariance.py` | 016 |
| `risk/stop_loss.py` | 015, 016 |
| `features/research_factors.py` | 012 |
| `features/options_factors.py` | 012 |
| `api/ab_testing.py` | 014 |

### Dependency order (implement in this sequence)

1. **006** Config Cleanup (removes dead flags, establishes config truth)
2. **007** Log Fixes + **008** Conceptual Fixes (clean foundation)
3. **005** Data Diagnostics (fix DATA_DIR import)
4. **012** Feature Engineering (fix lookahead bias before training)
5. **013** Model Training (fix CV and governance)
6. **009** Kelly Sizing (fix formula before paper trading)
7. **011** Backtest Realism (fix execution before validation)
8. **010** Health Check Rewrites (meaningful monitoring)
9. **015** Paper Trader Integration (wire all risk components)
10. **016** Risk System Improvements (better risk estimates)
11. **014** A/B Testing (test improvements properly)
12. **003** + **004** + **002** UI improvements
13. **017** System Innovation (advanced features)
14. **001** Statistical Jump Model (new regime detection)

---

## How to Use These Specs

Each spec follows the [SPEC_TEMPLATE.md](../SPEC_TEMPLATE.md) format:

1. **Read the spec** — understand Why, What, Constraints
2. **Pick a task** — each task is ≤3 files, ≤30 minutes
3. **Implement** — follow the implementation notes exactly
4. **Verify** — run the verification command at the bottom of each task
5. **Move to next task** — tasks within a spec can often be done in order

### Rules
- One task at a time, fresh context per task
- Never modify files not listed in the task
- If a task's verification fails, fix before moving on
- If implementation notes conflict with actual code, the actual code wins (update the spec)
