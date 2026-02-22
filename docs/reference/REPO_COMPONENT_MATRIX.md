# Repository Component Matrix

## Package-Level Summary

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

## Module Index by Package

## `(root)`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `__init__.py` | 6 | 0 | 0 | Quant Engine - Continuous Feature ML Trading System |
| `config.py` | 271 | 0 | 0 | Central configuration for the quant engine. |
| `reproducibility.py` | 333 | 0 | 6 | Reproducibility locks for run manifests. |
| `run_autopilot.py` | 88 | 0 | 1 | Run one full autopilot cycle: |
| `run_backtest.py` | 426 | 0 | 1 | Backtest the trained model on historical data. |
| `run_dash.py` | 82 | 0 | 2 | Quant Engine -- Dash Dashboard Launcher. |
| `run_kalshi_event_pipeline.py` | 225 | 0 | 2 | Run the integrated Kalshi event-time pipeline inside quant_engine. |
| `run_predict.py` | 201 | 0 | 1 | Generate predictions using trained ensemble model. |
| `run_rehydrate_cache_metadata.py` | 99 | 0 | 2 | Backfill cache metadata sidecars for existing OHLCV cache files. |
| `run_retrain.py` | 289 | 0 | 2 | Retrain the quant engine model — checks triggers and retrains if needed. |
| `run_train.py` | 194 | 0 | 1 | Train the regime-conditional ensemble model. |
| `run_wrds_daily_refresh.py` | 347 | 0 | 5 | Re-download all daily OHLCV data from WRDS CRSP to replace old cache files |

## `autopilot`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `autopilot/__init__.py` | 20 | 0 | 0 | Autopilot layer: discovery, promotion, and paper-trading orchestration. |
| `autopilot/engine.py` | 910 | 2 | 0 | End-to-end autopilot cycle: |
| `autopilot/paper_trader.py` | 432 | 1 | 0 | Stateful paper-trading engine for promoted strategies. |
| `autopilot/promotion_gate.py` | 230 | 2 | 0 | Promotion gate for deciding whether a discovered strategy is deployable. |
| `autopilot/registry.py` | 103 | 2 | 0 | Persistent strategy registry for promoted candidates. |
| `autopilot/strategy_discovery.py` | 75 | 2 | 0 | Strategy discovery for execution-layer parameter variants. |

## `backtest`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `backtest/__init__.py` | 0 | 0 | 0 | No module docstring. |
| `backtest/advanced_validation.py` | 551 | 5 | 6 | Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis. |
| `backtest/engine.py` | 1625 | 3 | 0 | Backtester — converts model predictions into simulated trades. |
| `backtest/execution.py` | 271 | 2 | 1 | Execution simulator with spread, market impact, and participation limits. |
| `backtest/optimal_execution.py` | 199 | 0 | 2 | Almgren-Chriss (2001) optimal execution model. |
| `backtest/validation.py` | 726 | 5 | 7 | Walk-forward validation and statistical tests. |

## `dash_ui`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `dash_ui/__init__.py` | 23 | 0 | 0 | Quantitative Trading Engine - Dash UI Package |
| `dash_ui/app.py` | 160 | 0 | 1 | Quant Engine Professional Dashboard — Dash Application Factory. |
| `dash_ui/components/__init__.py` | 34 | 0 | 0 | Reusable Dash components for the Quant Engine Dashboard. |
| `dash_ui/components/alert_banner.py` | 130 | 0 | 1 | Alert banner component for displaying system messages and warnings. |
| `dash_ui/components/chart_utils.py` | 645 | 0 | 12 | Plotly chart factory functions for the Quant Engine Dashboard. |
| `dash_ui/components/health_check_list.py` | 38 | 0 | 2 | Reusable health check display component. |
| `dash_ui/components/metric_card.py` | 46 | 0 | 1 | Reusable KPI metric card component. |
| `dash_ui/components/regime_badge.py` | 61 | 0 | 1 | Regime state badge component for displaying market regime indicators. |
| `dash_ui/components/sidebar.py` | 121 | 0 | 3 | Sidebar navigation component with active-state highlighting. |
| `dash_ui/components/status_bar.py` | 34 | 0 | 1 | Bottom status bar component. |
| `dash_ui/components/trade_table.py` | 226 | 0 | 1 | Styled DataTable component for displaying trades with conditional formatting. |
| `dash_ui/data/__init__.py` | 1 | 0 | 0 | Data loading and caching layer for the Dash UI. |
| `dash_ui/data/cache.py` | 91 | 0 | 2 | Caching layer for Dash UI data loading operations. |
| `dash_ui/data/loaders.py` | 726 | 2 | 18 | Data loading and computation functions for the Dash UI. |
| `dash_ui/pages/__init__.py` | 26 | 0 | 0 | Dash page modules for the Quant Engine Dashboard. |
| `dash_ui/pages/autopilot_kalshi.py` | 771 | 0 | 8 | Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets. |
| `dash_ui/pages/backtest_risk.py` | 787 | 0 | 5 | Backtest & Risk -- Equity curves, risk metrics, and trade analysis. |
| `dash_ui/pages/dashboard.py` | 1040 | 0 | 12 | Dashboard -- Portfolio Intelligence Overview. |
| `dash_ui/pages/data_explorer.py` | 795 | 0 | 12 | Data Explorer -- OHLCV visualization and quality analysis. |
| `dash_ui/pages/iv_surface.py` | 677 | 0 | 10 | IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling. |
| `dash_ui/pages/model_lab.py` | 919 | 0 | 7 | Model Lab -- Feature engineering, regime detection, and model training. |
| `dash_ui/pages/signal_desk.py` | 536 | 0 | 3 | Signal Desk -- Prediction generation and signal ranking. |
| `dash_ui/pages/sp_comparison.py` | 501 | 0 | 3 | S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation. |
| `dash_ui/pages/system_health.py` | 1081 | 0 | 14 | System Health Console -- comprehensive health assessment for the Quant Engine. |
| `dash_ui/server.py` | 52 | 0 | 0 | Entry point for the Quant Engine Dash application. |
| `dash_ui/theme.py` | 205 | 0 | 6 | Bloomberg-inspired dark theme for the Quant Engine Dash Dashboard. |

## `data`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `data/__init__.py` | 13 | 0 | 0 | Data subpackage — self-contained data loading, caching, WRDS, and survivorship. |
| `data/alternative.py` | 652 | 1 | 2 | Alternative data framework — WRDS-backed implementation. |
| `data/feature_store.py` | 308 | 1 | 0 | Point-in-time feature store for backtest acceleration. |
| `data/loader.py` | 707 | 0 | 13 | Data loader — self-contained data loading with multiple sources. |
| `data/local_cache.py` | 674 | 0 | 21 | Local data cache for daily OHLCV data. |
| `data/provider_base.py` | 12 | 1 | 0 | Shared provider protocol for pluggable data connectors. |
| `data/provider_registry.py` | 49 | 0 | 5 | Provider registry for unified data-provider access (WRDS, Kalshi, ...). |
| `data/quality.py` | 237 | 1 | 3 | Data quality checks for OHLCV time series. |
| `data/survivorship.py` | 928 | 8 | 5 | Survivorship Bias Controls (Tasks 112-117) |
| `data/wrds_provider.py` | 1584 | 1 | 6 | wrds_provider.py |

## `features`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `features/__init__.py` | 0 | 0 | 0 | No module docstring. |
| `features/harx_spillovers.py` | 242 | 0 | 3 | HARX Volatility Spillover features (Tier 6.1). |
| `features/intraday.py` | 193 | 0 | 1 | Intraday microstructure features from WRDS TAQmsec tick data. |
| `features/lob_features.py` | 311 | 0 | 5 | Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2). |
| `features/macro.py` | 243 | 1 | 1 | FRED macro indicator features for quant_engine. |
| `features/options_factors.py` | 119 | 0 | 4 | Option surface factor construction from OptionMetrics-enriched daily panels. |
| `features/pipeline.py` | 819 | 1 | 9 | Feature Pipeline — computes model features from OHLCV data. |
| `features/research_factors.py` | 976 | 1 | 19 | Research-derived factor construction for quant_engine. |
| `features/wave_flow.py` | 144 | 0 | 1 | Wave-Flow Decomposition for quant_engine. |

## `indicators`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `indicators/__init__.py` | 57 | 0 | 0 | Quant Engine Indicators — self-contained copy of the technical indicator library. |
| `indicators/indicators.py` | 2635 | 92 | 2 | Technical Indicator Library |

## `kalshi`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `kalshi/__init__.py` | 58 | 0 | 0 | Kalshi vertical for intraday event-market research. |
| `kalshi/client.py` | 630 | 5 | 1 | Kalshi API client with signed authentication, rate limiting, and endpoint routing. |
| `kalshi/disagreement.py` | 112 | 1 | 2 | Cross-market disagreement engine for Kalshi event features. |
| `kalshi/distribution.py` | 917 | 3 | 23 | Contract -> probability distribution builder for Kalshi markets. |
| `kalshi/events.py` | 511 | 2 | 11 | Event-time joins and as-of feature/label builders for Kalshi-driven research. |
| `kalshi/mapping_store.py` | 70 | 2 | 0 | Versioned event-to-market mapping persistence. |
| `kalshi/microstructure.py` | 126 | 1 | 2 | Market microstructure diagnostics for Kalshi event markets. |
| `kalshi/options.py` | 144 | 0 | 3 | OptionMetrics-style options reference features for Kalshi event disagreement. |
| `kalshi/pipeline.py` | 158 | 1 | 0 | Orchestration helpers for the Kalshi event-market vertical. |
| `kalshi/promotion.py` | 176 | 1 | 2 | Event-strategy promotion helpers for Kalshi walk-forward outputs. |
| `kalshi/provider.py` | 628 | 1 | 3 | Kalshi provider: ingestion + storage + feature-ready retrieval. |
| `kalshi/quality.py` | 203 | 2 | 5 | Quality scoring helpers for Kalshi event-distribution snapshots. |
| `kalshi/regimes.py` | 141 | 1 | 6 | Regime tagging for Kalshi event strategies. |
| `kalshi/router.py` | 95 | 2 | 0 | Routing helpers for live vs historical Kalshi endpoints. |
| `kalshi/storage.py` | 623 | 1 | 0 | Event-time storage layer for Kalshi + macro event research. |
| `kalshi/tests/__init__.py` | 1 | 0 | 0 | Kalshi package-local tests. |
| `kalshi/tests/test_bin_validity.py` | 105 | 1 | 0 | Bin overlap/gap detection test (Instructions I.3). |
| `kalshi/tests/test_distribution.py` | 36 | 1 | 0 | No module docstring. |
| `kalshi/tests/test_leakage.py` | 41 | 1 | 0 | No module docstring. |
| `kalshi/tests/test_no_leakage.py` | 117 | 1 | 0 | No-leakage test at panel level (Instructions I.4). |
| `kalshi/tests/test_signature_kat.py` | 141 | 1 | 0 | Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1). |
| `kalshi/tests/test_stale_quotes.py` | 152 | 1 | 0 | Stale quote cutoff test (Instructions I.5). |
| `kalshi/tests/test_threshold_direction.py` | 126 | 1 | 0 | Threshold direction correctness test (Instructions I.2). |
| `kalshi/tests/test_walkforward_purge.py` | 159 | 1 | 0 | Walk-forward purge/embargo test (Instructions I.6). |
| `kalshi/walkforward.py` | 477 | 3 | 8 | Walk-forward evaluation for event-centric Kalshi feature panels. |

## `models`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `models/__init__.py` | 20 | 0 | 0 | Models subpackage — training, prediction, versioning, and retraining triggers. |
| `models/calibration.py` | 216 | 2 | 0 | Confidence Calibration --- Platt scaling and isotonic regression. |
| `models/cross_sectional.py` | 136 | 0 | 1 | Cross-Sectional Ranking Model — rank stocks relative to peers at each date. |
| `models/feature_stability.py` | 311 | 2 | 0 | Feature Stability Monitoring — tracks feature importance rankings across |
| `models/governance.py` | 108 | 2 | 0 | Champion/challenger governance for model versions. |
| `models/iv/__init__.py` | 31 | 0 | 0 | Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface. |
| `models/iv/models.py` | 928 | 10 | 1 | Implied Volatility Surface Models. |
| `models/neural_net.py` | 197 | 1 | 0 | Tabular Neural Network — feedforward network for tabular financial data. |
| `models/predictor.py` | 375 | 1 | 1 | Model Predictor — loads trained ensemble and generates predictions. |
| `models/retrain_trigger.py` | 296 | 1 | 0 | ML Retraining Trigger Logic |
| `models/trainer.py` | 1340 | 5 | 0 | Model Trainer — trains regime-conditional gradient boosting ensemble. |
| `models/versioning.py` | 204 | 2 | 0 | Model Versioning — timestamped model directories with registry. |
| `models/walk_forward.py` | 235 | 0 | 4 | Walk-Forward Model Selection — expanding-window hyperparameter search |

## `regime`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `regime/__init__.py` | 14 | 0 | 0 | Regime modeling components. |
| `regime/correlation.py` | 212 | 1 | 0 | Correlation Regime Detection (NEW 11). |
| `regime/detector.py` | 292 | 2 | 1 | Regime detector with two engines: |
| `regime/hmm.py` | 501 | 2 | 4 | Gaussian HMM regime model with sticky transitions and duration smoothing. |

## `risk`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `risk/__init__.py` | 42 | 0 | 0 | Risk Management Module — Renaissance-grade portfolio risk controls. |
| `risk/attribution.py` | 266 | 0 | 4 | Performance Attribution --- decompose portfolio returns into market, factor, and alpha. |
| `risk/covariance.py` | 244 | 2 | 2 | Covariance estimation utilities for portfolio risk controls. |
| `risk/drawdown.py` | 233 | 3 | 0 | Drawdown Controller — circuit breakers and recovery protocols. |
| `risk/factor_portfolio.py` | 220 | 0 | 2 | Factor-Based Portfolio Construction — factor decomposition and exposure analysis. |
| `risk/metrics.py` | 251 | 2 | 0 | Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics. |
| `risk/portfolio_optimizer.py` | 255 | 0 | 1 | Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction. |
| `risk/portfolio_risk.py` | 327 | 2 | 0 | Portfolio Risk Manager — enforces sector, correlation, and exposure limits. |
| `risk/position_sizer.py` | 290 | 2 | 0 | Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods. |
| `risk/stop_loss.py` | 251 | 3 | 0 | Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops. |
| `risk/stress_test.py` | 363 | 0 | 5 | Stress Testing Module --- scenario analysis and historical drawdown replay. |

## `tests`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `tests/__init__.py` | 2 | 0 | 0 | No module docstring. |
| `tests/test_autopilot_predictor_fallback.py` | 53 | 1 | 0 | No module docstring. |
| `tests/test_cache_metadata_rehydrate.py` | 91 | 1 | 1 | No module docstring. |
| `tests/test_covariance_estimator.py` | 20 | 1 | 0 | No module docstring. |
| `tests/test_delisting_total_return.py` | 71 | 1 | 0 | No module docstring. |
| `tests/test_drawdown_liquidation.py` | 128 | 6 | 0 | No module docstring. |
| `tests/test_execution_dynamic_costs.py` | 43 | 1 | 0 | No module docstring. |
| `tests/test_integration.py` | 556 | 4 | 1 | End-to-end integration tests for the quant engine pipeline. |
| `tests/test_iv_arbitrage_builder.py` | 33 | 1 | 0 | No module docstring. |
| `tests/test_kalshi_asof_features.py` | 60 | 1 | 0 | No module docstring. |
| `tests/test_kalshi_distribution.py` | 109 | 1 | 0 | No module docstring. |
| `tests/test_kalshi_hardening.py` | 600 | 1 | 0 | No module docstring. |
| `tests/test_loader_and_predictor.py` | 220 | 3 | 0 | No module docstring. |
| `tests/test_panel_split.py` | 50 | 1 | 0 | No module docstring. |
| `tests/test_paper_trader_kelly.py` | 120 | 1 | 3 | No module docstring. |
| `tests/test_promotion_contract.py` | 95 | 1 | 2 | No module docstring. |
| `tests/test_provider_registry.py` | 24 | 1 | 0 | No module docstring. |
| `tests/test_research_factors.py` | 127 | 1 | 1 | No module docstring. |
| `tests/test_survivorship_pit.py` | 69 | 1 | 0 | No module docstring. |
| `tests/test_validation_and_risk_extensions.py` | 101 | 1 | 1 | No module docstring. |

## `utils`
| Module | Lines | Classes | Top-Level Functions | Module Intent |
|---|---:|---:|---:|---|
| `utils/__init__.py` | 1 | 0 | 0 | No module docstring. |
| `utils/logging.py` | 436 | 3 | 1 | Structured logging for the quant engine. |
