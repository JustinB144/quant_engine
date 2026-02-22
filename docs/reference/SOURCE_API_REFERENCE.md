# Source API Reference

This is a source-derived inventory of modules, classes, methods, and top-level functions for the current repository state.

Use this for lookups and scoping edits. For architecture intent, read `../reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md`.

## Package `(root)`

### `__init__.py`
- Lines: 6
- Module intent: Quant Engine - Continuous Feature ML Trading System
- Imports: none
- Classes: none
- Top-level functions: none

### `config.py`
- Lines: 271
- Module intent: Central configuration for the quant engine.
- Imports (2): `pathlib`, `typing`
- Classes: none
- Top-level functions: none

### `reproducibility.py`
- Lines: 333
- Module intent: Reproducibility locks for run manifests.
- Imports (8): `__future__`, `hashlib`, `json`, `subprocess`, `datetime`, `pathlib`, `typing`, `pandas`
- Classes: none
- Top-level functions:
  - `_get_git_commit()` (line 22): Return the current git commit hash, or 'unknown' if not in a repo.
  - `_dataframe_checksum(df)` (line 36): Compute a lightweight checksum of a DataFrame's shape and sample.
  - `build_run_manifest(run_type, config_snapshot, datasets, mapping_version, extra)` (line 47): Build a reproducibility manifest for a pipeline run.
  - `write_run_manifest(manifest, output_dir, filename)` (line 99): Write manifest to JSON file. Returns the output path.
  - `verify_manifest(manifest_path, config_snapshot)` (line 117): Verify current environment matches a stored manifest.
  - `replay_manifest(manifest_path, output_dir)` (line 234): Re-run a historical cycle and compare to stored results.

### `run_autopilot.py`
- Lines: 88
- Module intent: Run one full autopilot cycle:
- Imports (6): `argparse`, `sys`, `time`, `pathlib`, `quant_engine.autopilot.engine`, `quant_engine.config`
- Classes: none
- Top-level functions:
  - `main()` (line 23): No function docstring.

### `run_backtest.py`
- Lines: 426
- Module intent: Backtest the trained model on historical data.
- Imports (16): `argparse`, `json`, `logging`, `sys`, `time`, `pathlib`, `numpy`, `pandas`, `quant_engine.config`, `quant_engine.data.loader`, `quant_engine.data.survivorship`, `quant_engine.features.pipeline`, `quant_engine.regime.detector`, `quant_engine.models.predictor`, `quant_engine.backtest.engine`, `quant_engine.backtest.validation`
- Classes: none
- Top-level functions:
  - `main()` (line 50): No function docstring.

### `run_dash.py`
- Lines: 82
- Module intent: Quant Engine -- Dash Dashboard Launcher.
- Imports (3): `argparse`, `sys`, `pathlib`
- Classes: none
- Top-level functions:
  - `_check_dependencies()` (line 26): Return list of missing package install names.
  - `main()` (line 37): Launch the Quant Engine Dash Dashboard.

### `run_kalshi_event_pipeline.py`
- Lines: 225
- Module intent: Run the integrated Kalshi event-time pipeline inside quant_engine.
- Imports (10): `argparse`, `json`, `pathlib`, `pandas`, `quant_engine.config`, `quant_engine.kalshi.distribution`, `quant_engine.kalshi.events`, `quant_engine.kalshi.pipeline`, `quant_engine.kalshi.promotion`, `quant_engine.kalshi.walkforward`
- Classes: none
- Top-level functions:
  - `_read_df(path)` (line 37): No function docstring.
  - `main()` (line 44): No function docstring.

### `run_predict.py`
- Lines: 201
- Module intent: Generate predictions using trained ensemble model.
- Imports (12): `argparse`, `json`, `sys`, `time`, `pathlib`, `numpy`, `pandas`, `quant_engine.config`, `quant_engine.data.loader`, `quant_engine.features.pipeline`, `quant_engine.regime.detector`, `quant_engine.models.predictor`
- Classes: none
- Top-level functions:
  - `main()` (line 33): No function docstring.

### `run_rehydrate_cache_metadata.py`
- Lines: 99
- Module intent: Backfill cache metadata sidecars for existing OHLCV cache files.
- Imports (5): `argparse`, `sys`, `pathlib`, `quant_engine.config`, `quant_engine.data.local_cache`
- Classes: none
- Top-level functions:
  - `_parse_root_source(items)` (line 21): No function docstring.
  - `main()` (line 35): No function docstring.

### `run_retrain.py`
- Lines: 289
- Module intent: Retrain the quant engine model — checks triggers and retrains if needed.
- Imports (14): `argparse`, `sys`, `time`, `pathlib`, `numpy`, `pandas`, `quant_engine.config`, `quant_engine.data.loader`, `quant_engine.features.pipeline`, `quant_engine.regime.detector`, `quant_engine.models.governance`, `quant_engine.models.trainer`, `quant_engine.models.retrain_trigger`, `quant_engine.models.versioning`
- Classes: none
- Top-level functions:
  - `_check_regime_change_trigger(predictions_df, trained_regime, days_threshold)` (line 36): Check whether the market regime has changed for a sustained period.
  - `main()` (line 67): No function docstring.

### `run_train.py`
- Lines: 194
- Module intent: Train the regime-conditional ensemble model.
- Imports (13): `argparse`, `sys`, `time`, `pathlib`, `numpy`, `pandas`, `quant_engine.config`, `quant_engine.data.loader`, `quant_engine.features.pipeline`, `quant_engine.regime.detector`, `quant_engine.models.governance`, `quant_engine.models.trainer`, `quant_engine.models.versioning`
- Classes: none
- Top-level functions:
  - `main()` (line 35): No function docstring.

### `run_wrds_daily_refresh.py`
- Lines: 347
- Module intent: Re-download all daily OHLCV data from WRDS CRSP to replace old cache files
- Imports (8): `argparse`, `sys`, `time`, `datetime`, `pathlib`, `pandas`, `quant_engine.config`, `quant_engine.data.local_cache`
- Classes: none
- Top-level functions:
  - `_build_ticker_list(tickers_arg)` (line 30): Build the full ticker list from cached + UNIVERSE_FULL + BENCHMARK.
  - `_verify_file(path)` (line 41): Verify OHLCV quality for a single parquet file. Returns dict of results.
  - `_verify_all(cache_dir)` (line 93): Run verification on all _1d.parquet files in cache.
  - `_cleanup_old_daily(cache_dir, downloaded_tickers)` (line 148): Remove old {TICKER}_daily_{dates}.parquet and .meta.json files.
  - `main()` (line 169): No function docstring.

## Package `autopilot`

### `autopilot/__init__.py`
- Lines: 20
- Module intent: Autopilot layer: discovery, promotion, and paper-trading orchestration.
- Imports (5): `.strategy_discovery`, `.promotion_gate`, `.registry`, `.paper_trader`, `.engine`
- Classes: none
- Top-level functions: none

### `autopilot/engine.py`
- Lines: 910
- Module intent: End-to-end autopilot cycle:
- Imports (24): `json`, `pathlib`, `typing`, `re`, `numpy`, `pandas`, `..backtest.engine`, `..backtest.advanced_validation`, `..backtest.validation`, `..models.walk_forward`, `..config`, `..data.loader`, `..data.survivorship`, `..features.pipeline`, `..models.cross_sectional`, `..models.predictor`, `..models.trainer`, `..regime.detector`, `..risk.covariance`, `..risk.portfolio_optimizer`, `.paper_trader`, `.promotion_gate`, `.registry`, `.strategy_discovery`
- Classes:
  - `HeuristicPredictor` (line 50): Lightweight fallback predictor used when sklearn-backed model artifacts
    - `__init__(self, horizon)` (line 56): No method docstring.
    - `_rolling_zscore(series, window, min_periods)` (line 61): No method docstring.
    - `predict(self, features, regimes, regime_confidence, regime_probabilities)` (line 66): No method docstring.
  - `AutopilotEngine` (line 122): Coordinates discovery, promotion, and paper execution.
    - `__init__(self, tickers, horizon, years, feature_mode, model_version, max_candidates, strict_oos, survivorship_mode, walk_forward, verbose, report_path)` (line 125): No method docstring.
    - `_log(self, msg)` (line 157): No method docstring.
    - `_is_permno_key(value)` (line 162): No method docstring.
    - `_assert_permno_price_data(self, data, context)` (line 165): No method docstring.
    - `_assert_permno_prediction_panel(self, panel, context)` (line 173): No method docstring.
    - `_assert_permno_latest_predictions(self, latest, context)` (line 183): No method docstring.
    - `_load_data(self)` (line 193): No method docstring.
    - `_build_regimes(self, features, data)` (line 205): No method docstring.
    - `_train_baseline(self, data)` (line 226): Train on the earliest 80% of dates so the most recent 20% remains OOS.
    - `_ensure_predictor(self, data)` (line 261): No method docstring.
    - `_predict_universe(self, data, predictor)` (line 287): No method docstring.
    - `_walk_forward_predictions(self, data)` (line 388): Generate OOS predictions via rolling walk-forward training.
    - `_evaluate_candidates(self, candidates, predictions, price_data)` (line 548): No method docstring.
    - `_compute_optimizer_weights(self, latest_predictions, data)` (line 720): Compute confidence-weighted portfolio optimizer weights.
    - `run_cycle(self)` (line 834): No method docstring.
- Top-level functions: none

### `autopilot/paper_trader.py`
- Lines: 432
- Module intent: Stateful paper-trading engine for promoted strategies.
- Imports (9): `json`, `datetime`, `pathlib`, `typing`, `numpy`, `pandas`, `..config`, `..risk.position_sizer`, `.registry`
- Classes:
  - `PaperTrader` (line 28): Executes paper entries/exits from promoted strategy definitions.
    - `__init__(self, state_path, initial_capital, max_total_positions, transaction_cost_bps, use_kelly_sizing, kelly_fraction, kelly_lookback_trades, kelly_min_size_multiplier, kelly_max_size_multiplier)` (line 33): No method docstring.
    - `_load_state(self)` (line 56): No method docstring.
    - `_save_state(self, state)` (line 68): No method docstring.
    - `_resolve_as_of(price_data)` (line 75): No method docstring.
    - `_latest_predictions_by_id(latest_predictions)` (line 85): No method docstring.
    - `_latest_predictions_by_ticker(latest_predictions)` (line 98): No method docstring.
    - `_current_price(ticker, as_of, price_data)` (line 103): No method docstring.
    - `_position_id(position)` (line 121): Canonical position key (PERMNO in PERMNO-first mode), with legacy fallback.
    - `_mark_to_market(self, state, as_of, price_data)` (line 129): No method docstring.
    - `_trade_return(trade)` (line 138): No method docstring.
    - `_historical_trade_stats(self, state, strategy_id)` (line 147): Estimate win/loss stats from recent closed paper trades.
    - `_market_risk_stats(ticker, as_of, price_data)` (line 183): No method docstring.
    - `_position_size_pct(self, state, strategy_id, base_position_size_pct, max_holding_days, confidence, regime, ticker, as_of, price_data)` (line 220): Compute entry size percentage. Uses bounded fractional-Kelly when enabled.
    - `run_cycle(self, active_strategies, latest_predictions, price_data, as_of)` (line 270): No method docstring.
- Top-level functions: none

### `autopilot/promotion_gate.py`
- Lines: 230
- Module intent: Promotion gate for deciding whether a discovered strategy is deployable.
- Imports (6): `dataclasses`, `typing`, `numpy`, `..backtest.engine`, `..config`, `.strategy_discovery`
- Classes:
  - `PromotionDecision` (line 34): No class docstring.
    - `to_dict(self)` (line 41): No method docstring.
  - `PromotionGate` (line 47): Applies hard risk/quality constraints before a strategy can be paper-deployed.
    - `__init__(self, min_trades, min_win_rate, min_sharpe, min_profit_factor, max_drawdown, min_annual_return, require_advanced_contract, max_dsr_pvalue, max_pbo, require_capacity_unconstrained, max_capacity_utilization, min_wf_oos_corr, min_wf_positive_fold_fraction, max_wf_is_oos_gap, min_regime_positive_fraction, event_max_worst_event_loss, event_min_surprise_hit_rate, event_min_regime_stability)` (line 52): No method docstring.
    - `evaluate(self, candidate, result, contract_metrics, event_mode)` (line 92): No method docstring.
    - `evaluate_event_strategy(self, candidate, result, event_metrics, contract_metrics)` (line 209): Evaluate an event strategy using standard + event-specific contract checks.
    - `rank(decisions)` (line 229): No method docstring.
- Top-level functions: none

### `autopilot/registry.py`
- Lines: 103
- Module intent: Persistent strategy registry for promoted candidates.
- Imports (7): `json`, `dataclasses`, `datetime`, `pathlib`, `typing`, `..config`, `.promotion_gate`
- Classes:
  - `ActiveStrategy` (line 15): No class docstring.
    - `to_dict(self)` (line 23): No method docstring.
  - `StrategyRegistry` (line 27): Maintains promoted strategy state and historical promotion decisions.
    - `__init__(self, path)` (line 32): No method docstring.
    - `_load(self)` (line 36): No method docstring.
    - `_save(self, payload)` (line 42): No method docstring.
    - `get_active(self)` (line 46): No method docstring.
    - `apply_promotions(self, decisions, max_active)` (line 50): No method docstring.
- Top-level functions: none

### `autopilot/strategy_discovery.py`
- Lines: 75
- Module intent: Strategy discovery for execution-layer parameter variants.
- Imports (3): `dataclasses`, `typing`, `..config`
- Classes:
  - `StrategyCandidate` (line 22): No class docstring.
    - `to_dict(self)` (line 31): No method docstring.
  - `StrategyDiscovery` (line 35): Generates a deterministic candidate grid for backtest validation.
    - `__init__(self, base_entry_threshold, base_confidence_threshold, base_position_size_pct)` (line 38): No method docstring.
    - `generate(self, horizon)` (line 48): No method docstring.
- Top-level functions: none

## Package `backtest`

### `backtest/__init__.py`
- Lines: 0
- Module intent: No module docstring.
- Imports: none
- Classes: none
- Top-level functions: none

### `backtest/advanced_validation.py`
- Lines: 551
- Module intent: Advanced Validation — Deflated Sharpe, PBO, Monte Carlo, capacity analysis.
- Imports (5): `dataclasses`, `math`, `typing`, `numpy`, `pandas`
- Classes:
  - `DeflatedSharpeResult` (line 37): Result of Deflated Sharpe Ratio test.
    - Methods: none
  - `PBOResult` (line 48): Probability of Backtest Overfitting result.
    - Methods: none
  - `MonteCarloResult` (line 58): Monte Carlo simulation result.
    - Methods: none
  - `CapacityResult` (line 72): Strategy capacity analysis.
    - Methods: none
  - `AdvancedValidationReport` (line 83): Complete advanced validation report.
    - Methods: none
- Top-level functions:
  - `deflated_sharpe_ratio(observed_sharpe, n_trials, n_returns, skewness, kurtosis, annualization_factor)` (line 93): Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
  - `probability_of_backtest_overfitting(returns_matrix, n_partitions)` (line 162): Probability of Backtest Overfitting (Bailey et al., 2017).
  - `monte_carlo_validation(trade_returns, n_simulations, holding_days, method)` (line 261): Monte Carlo validation of strategy performance.
  - `capacity_analysis(trades, price_data, capital_usd, max_participation_rate, impact_coefficient_bps)` (line 337): Estimate strategy capacity and market impact.
  - `run_advanced_validation(trade_returns, trades, price_data, n_strategy_variants, holding_days, returns_matrix, verbose)` (line 430): Run all advanced validation tests.
  - `_print_report(report)` (line 511): Pretty-print advanced validation report.

### `backtest/engine.py`
- Lines: 1625
- Module intent: Backtester — converts model predictions into simulated trades.
- Imports (9): `dataclasses`, `datetime`, `typing`, `re`, `numpy`, `pandas`, `logging`, `..config`, `.execution`
- Classes:
  - `Trade` (line 46): No class docstring.
    - Methods: none
  - `BacktestResult` (line 68): No class docstring.
    - Methods: none
  - `Backtester` (line 101): Simulates trading from model predictions.
    - `__init__(self, entry_threshold, confidence_threshold, transaction_cost_bps, holding_days, max_positions, position_size_pct, slippage_pct, use_risk_management, assumed_capital_usd, spread_bps, max_participation_rate, impact_coefficient_bps, min_fill_ratio)` (line 118): No method docstring.
    - `_init_risk_components(self)` (line 171): Initialize risk management components.
    - `_almgren_chriss_cost_bps(self, shares, reference_price, daily_volume, daily_volatility, n_intervals)` (line 192): Compute Almgren-Chriss execution cost in basis points.
    - `_simulate_entry(self, ohlcv, entry_idx, position_size)` (line 230): Simulate entry execution with participation and impact constraints.
    - `_simulate_exit(self, ohlcv, exit_idx, shares, force_full)` (line 297): Simulate exit execution, upgrading to Almgren-Chriss for large positions.
    - `_execution_context(ohlcv, bar_idx)` (line 358): Build local microstructure context for conditional execution costs.
    - `_effective_return_series(ohlcv)` (line 391): Return the best available close-to-close return stream.
    - `_delisting_adjustment_multiplier(self, ohlcv, entry_idx, exit_idx)` (line 411): Convert ret-based price-path return into total-return-path return.
    - `_trade_realized_return(self, ohlcv, entry_idx, exit_idx, entry_price, exit_price)` (line 452): Trade-level return including any total-return delisting adjustments.
    - `_is_permno_key(value)` (line 472): No method docstring.
    - `_assert_permno_inputs(self, predictions, price_data)` (line 475): Enforce PERMNO-keyed panels at runtime when strict identity mode is on.
    - `run(self, predictions, price_data, verbose)` (line 504): Run backtest across all tickers with cross-ticker position limits.
    - `_process_signals(self, signals_df, price_data)` (line 591): Process signals by day with capacity-aware ranking and realistic execution.
    - `_process_signals_risk_managed(self, signals_df, price_data)` (line 699): Process signals with full risk management.
    - `_compute_metrics(self, trades, price_data, verbose)` (line 1067): Compute performance metrics from trade list.
    - `_build_daily_equity(self, trades, price_data)` (line 1241): Build a proper time-weighted daily equity curve.
    - `_compute_turnover(self, trades)` (line 1304): Compute realized portfolio turnover from the trade list.
    - `_compute_regime_performance(self, trades)` (line 1368): Compute detailed per-regime performance metrics.
    - `_compute_tca(self, trades)` (line 1453): Compute Transaction Cost Analysis (TCA) report.
    - `_print_result(self, r)` (line 1527): Print backtest results.
    - `_empty_result(self)` (line 1618): No method docstring.
- Top-level functions: none

### `backtest/execution.py`
- Lines: 271
- Module intent: Execution simulator with spread, market impact, and participation limits.
- Imports (5): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `ExecutionFill` (line 17): No class docstring.
    - Methods: none
  - `ExecutionModel` (line 26): Simple market-impact model for backtests.
    - `__init__(self, spread_bps, max_participation_rate, impact_coefficient_bps, min_fill_ratio, dynamic_costs, dollar_volume_ref_usd, vol_ref, vol_spread_beta, gap_spread_beta, range_spread_beta, vol_impact_beta)` (line 31): No method docstring.
    - `simulate(self, side, reference_price, daily_volume, desired_notional_usd, force_full, realized_vol, overnight_gap, intraday_range, event_spread_multiplier)` (line 57): Simulate execution against daily volume capacity.
- Top-level functions:
  - `calibrate_cost_model(fills, actual_prices)` (line 151): Calibrate execution-cost model parameters from historical fills.

### `backtest/optimal_execution.py`
- Lines: 199
- Module intent: Almgren-Chriss (2001) optimal execution model.
- Imports (2): `__future__`, `numpy`
- Classes: none
- Top-level functions:
  - `almgren_chriss_trajectory(total_shares, n_intervals, daily_volume, daily_volatility, risk_aversion, temporary_impact, permanent_impact)` (line 18): Compute the optimal execution trajectory using the Almgren-Chriss model.
  - `estimate_execution_cost(trajectory, reference_price, daily_volume, daily_volatility, temporary_impact, permanent_impact)` (line 104): Estimate the total execution cost for a given trade trajectory.

### `backtest/validation.py`
- Lines: 726
- Module intent: Walk-forward validation and statistical tests.
- Imports (6): `dataclasses`, `itertools`, `math`, `typing`, `numpy`, `pandas`
- Classes:
  - `WalkForwardFold` (line 104): No class docstring.
    - Methods: none
  - `WalkForwardResult` (line 114): No class docstring.
    - Methods: none
  - `StatisticalTests` (line 127): No class docstring.
    - Methods: none
  - `CPCVResult` (line 153): No class docstring.
    - Methods: none
  - `SPAResult` (line 169): No class docstring.
    - Methods: none
- Top-level functions:
  - `walk_forward_validate(predictions, actuals, n_folds, entry_threshold, max_overfit_ratio, purge_gap, embargo)` (line 180): Walk-forward validation of prediction quality with purge gap.
  - `_benjamini_hochberg(pvals, alpha)` (line 314): Benjamini-Hochberg procedure for multiple testing correction.
  - `run_statistical_tests(predictions, actuals, trade_returns, entry_threshold, holding_days)` (line 338): Statistical tests for prediction quality.
  - `_partition_bounds(n_obs, n_partitions)` (line 477): Return contiguous [start, end) bounds for temporal partitions.
  - `combinatorial_purged_cv(predictions, actuals, entry_threshold, n_partitions, n_test_partitions, purge_gap, embargo, max_combinations)` (line 492): Combinatorial Purged Cross-Validation for signal robustness.
  - `strategy_signal_returns(predictions, actuals, entry_threshold, confidence, min_confidence)` (line 635): Build per-sample strategy return series from prediction signals.
  - `superior_predictive_ability(strategy_returns, benchmark_returns, n_bootstraps, block_size, random_state)` (line 661): Single-strategy SPA-style block-bootstrap test on differential returns.

## Package `dash_ui`

### `dash_ui/__init__.py`
- Lines: 23
- Module intent: Quantitative Trading Engine - Dash UI Package
- Imports: none
- Classes: none
- Top-level functions: none

### `dash_ui/app.py`
- Lines: 160
- Module intent: Quant Engine Professional Dashboard — Dash Application Factory.
- Imports (5): `dash`, `dash_bootstrap_components`, `dash`, `.theme`, `.components.sidebar`
- Classes: none
- Top-level functions:
  - `create_app()` (line 23): Create and configure the Dash application with multi-page support.

### `dash_ui/components/__init__.py`
- Lines: 34
- Module intent: Reusable Dash components for the Quant Engine Dashboard.
- Imports (6): `.metric_card`, `.trade_table`, `.regime_badge`, `.alert_banner`, `.sidebar`, `.`
- Classes: none
- Top-level functions: none

### `dash_ui/components/alert_banner.py`
- Lines: 130
- Module intent: Alert banner component for displaying system messages and warnings.
- Imports (4): `typing`, `dash_bootstrap_components`, `dash`, `..theme`
- Classes: none
- Top-level functions:
  - `alert_banner(message, severity, icon, dismissable)` (line 26): Create a styled alert banner message.

### `dash_ui/components/chart_utils.py`
- Lines: 645
- Module intent: Plotly chart factory functions for the Quant Engine Dashboard.
- Imports (6): `typing`, `numpy`, `pandas`, `plotly.graph_objects`, `plotly.subplots`, `..theme`
- Classes: none
- Top-level functions:
  - `line_chart(data_dict, title, xaxis_title, yaxis_title, kwargs)` (line 42): Create a multi-line chart.
  - `area_chart(x, y, label, color, title, kwargs)` (line 86): Create a filled area chart.
  - `bar_chart(labels, values, title, colors, horizontal, kwargs)` (line 127): Create a bar chart (vertical or horizontal).
  - `heatmap_chart(z, x_labels, y_labels, title, colorscale, fmt, kwargs)` (line 173): Create an annotated heatmap (e.g., correlation matrix).
  - `surface_3d(X, Y, Z, title, colorscale, kwargs)` (line 223): Create a 3D surface plot.
  - `equity_curve(dates, equity, benchmark, title, kwargs)` (line 266): Create an equity curve chart with drawdown shading.
  - `regime_timeline(dates, regimes, regime_names, title, kwargs)` (line 324): Create a regime timeline with colored vertical bands.
  - `dual_axis_chart(x, y1, y2, label1, label2, title, kwargs)` (line 390): Create a chart with two y-axes (left and right).
  - `candlestick_chart(df, title, kwargs)` (line 444): Create a candlestick chart with volume subplot.
  - `scatter_chart(x, y, color, size, text, title, xaxis_title, yaxis_title, kwargs)` (line 505): Create a scatter plot with optional color/size encoding.
  - `radar_chart(categories, values, name, title, kwargs)` (line 553): Create a radar/polar chart.
  - `histogram_chart(values, nbins, title, color, kwargs)` (line 596): Create a distribution histogram.

### `dash_ui/components/health_check_list.py`
- Lines: 38
- Module intent: Reusable health check display component.
- Imports (1): `dash`
- Classes: none
- Top-level functions:
  - `health_check_item(label, status, detail)` (line 8): Render a single health check row with icon, label, and optional detail.
  - `health_check_list(checks)` (line 29): Render a list of health checks.

### `dash_ui/components/metric_card.py`
- Lines: 46
- Module intent: Reusable KPI metric card component.
- Imports (1): `dash`
- Classes: none
- Top-level functions:
  - `metric_card(label, value, color, subtitle, icon)` (line 5): Create a styled KPI metric card.

### `dash_ui/components/regime_badge.py`
- Lines: 61
- Module intent: Regime state badge component for displaying market regime indicators.
- Imports (2): `dash`, `..theme`
- Classes: none
- Top-level functions:
  - `regime_badge(regime_code)` (line 12): Create a colored badge displaying the regime name.

### `dash_ui/components/sidebar.py`
- Lines: 121
- Module intent: Sidebar navigation component with active-state highlighting.
- Imports (2): `dash`, `dash`
- Classes: none
- Top-level functions:
  - `_nav_item(item)` (line 18): Render a single navigation item.
  - `create_sidebar()` (line 39): Create the full sidebar layout.
  - `update_active_nav(pathname)` (line 109): Highlight the active navigation item based on URL.

### `dash_ui/components/status_bar.py`
- Lines: 34
- Module intent: Bottom status bar component.
- Imports (3): `sys`, `datetime`, `dash`
- Classes: none
- Top-level functions:
  - `create_status_bar()` (line 7): Create the bottom status bar.

### `dash_ui/components/trade_table.py`
- Lines: 226
- Module intent: Styled DataTable component for displaying trades with conditional formatting.
- Imports (3): `typing`, `dash`, `..theme`
- Classes: none
- Top-level functions:
  - `trade_table(id, columns, data)` (line 18): Create a styled DataTable for displaying trade-level data.

### `dash_ui/data/__init__.py`
- Lines: 1
- Module intent: Data loading and caching layer for the Dash UI.
- Imports: none
- Classes: none
- Top-level functions: none

### `dash_ui/data/cache.py`
- Lines: 91
- Module intent: Caching layer for Dash UI data loading operations.
- Imports (5): `tempfile`, `functools`, `pathlib`, `typing`, `flask_caching`
- Classes: none
- Top-level functions:
  - `init_cache(app)` (line 33): Initialize the Flask cache instance with a Dash application.
  - `cached(timeout)` (line 53): Decorator to cache function results for a specified duration.

### `dash_ui/data/loaders.py`
- Lines: 726
- Module intent: Data loading and computation functions for the Dash UI.
- Imports (8): `__future__`, `json`, `dataclasses`, `datetime`, `pathlib`, `typing`, `numpy`, `pandas`
- Classes:
  - `HealthCheck` (line 369): Single health check result.
    - Methods: none
  - `SystemHealthPayload` (line 379): Full system health assessment.
    - Methods: none
- Top-level functions:
  - `load_trades(path)` (line 21): Load and clean backtest trade CSV.
  - `build_portfolio_returns(trades)` (line 38): Build daily portfolio returns from trade-level data.
  - `_read_close_returns(path)` (line 55): Read close returns from a parquet file.
  - `load_benchmark_returns(cache_dir, ref_index)` (line 64): Load benchmark (SPY) returns from parquet cache.
  - `load_factor_proxies(cache_dir, ref_index)` (line 113): Load factor proxy returns for attribution.
  - `compute_risk_metrics(returns)` (line 150): Compute portfolio risk metrics from daily returns.
  - `compute_regime_payload(cache_dir)` (line 183): Run HMM regime detection and return structured results.
  - `compute_model_health(model_dir, trades)` (line 247): Assess model health from registry and trade data.
  - `load_feature_importance(model_dir)` (line 300): Load feature importance from latest model metadata.
  - `compute_health_scores(model_health, trades, cv_gap)` (line 327): Quick health indicators for dashboard cards.
  - `score_to_status(score)` (line 403): Convert numeric score to PASS/WARN/FAIL status.
  - `collect_health_data()` (line 412): Run full system health assessment.
  - `_check_data_integrity()` (line 449): Check survivorship bias and data quality.
  - `_check_promotion_contract()` (line 512): Verify promotion gate configuration.
  - `_check_walkforward()` (line 561): Verify walk-forward validation setup.
  - `_check_execution()` (line 596): Audit execution cost model.
  - `_check_complexity()` (line 643): Audit feature and knob complexity.
  - `_check_strengths()` (line 696): Identify what's working well.

### `dash_ui/pages/__init__.py`
- Lines: 26
- Module intent: Dash page modules for the Quant Engine Dashboard.
- Imports: none
- Classes: none
- Top-level functions: none

### `dash_ui/pages/autopilot_kalshi.py`
- Lines: 771
- Module intent: Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets.
- Imports (11): `json`, `datetime`, `pathlib`, `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `quant_engine.dash_ui.components.metric_card`, `quant_engine.dash_ui.theme`
- Classes: none
- Top-level functions:
  - `_demo_strategy_candidates()` (line 55): Generate realistic demo strategy candidate data.
  - `_demo_promotion_funnel()` (line 92): Demo promotion funnel counts.
  - `_demo_paper_equity()` (line 104): Generate demo paper trading equity curve.
  - `_demo_paper_positions()` (line 114): Generate demo paper trading positions.
  - `_demo_kalshi_events(event_type)` (line 137): Generate demo Kalshi event data for a given type.
  - `update_autopilot(discover_clicks, registry_clicks)` (line 424): Run discovery cycle or load registry.
  - `load_paper_trading(n_clicks)` (line 520): Load paper trading state or use demo data.
  - `load_kalshi_events(n_clicks, event_type)` (line 596): Load Kalshi event data or use demo data.

### `dash_ui/pages/backtest_risk.py`
- Lines: 787
- Module intent: Backtest & Risk -- Equity curves, risk metrics, and trade analysis.
- Imports (10): `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `plotly.subplots`, `quant_engine.dash_ui.components.metric_card`, `quant_engine.dash_ui.theme`, `quant_engine.dash_ui.data.loaders`
- Classes: none
- Top-level functions:
  - `_control_group(label, component)` (line 69): No function docstring.
  - `_metrics_row()` (line 163): No function docstring.
  - `run_backtest(n_clicks, holding_period, max_positions, entry_thresh, pos_size, risk_mgmt)` (line 353): Load existing backtest results or run a new backtest.
  - `update_risk_analytics(returns_data)` (line 554): Compute and display risk analytics from stored returns.
  - `update_trade_analysis(trades_data)` (line 712): Populate trade table and regime performance chart.

### `dash_ui/pages/dashboard.py`
- Lines: 1040
- Module intent: Dashboard -- Portfolio Intelligence Overview.
- Imports (16): `__future__`, `json`, `traceback`, `datetime`, `pathlib`, `typing`, `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `plotly.subplots`, `quant_engine.dash_ui.components.metric_card`, `quant_engine.dash_ui.data.loaders`, `quant_engine.dash_ui.theme`
- Classes: none
- Top-level functions:
  - `_card_panel(title, children, extra_style)` (line 70): Wrap children inside a styled card-panel div.
  - `_pct(val, decimals)` (line 85): No function docstring.
  - `_fmt(val, decimals)` (line 89): No function docstring.
  - `load_dashboard_data(n_intervals, n_clicks)` (line 175): Load all data and cache in dcc.Store as JSON-safe dict.
  - `update_metric_cards(data)` (line 400): No function docstring.
  - `render_tab_content(tab, data)` (line 456): No function docstring.
  - `_render_portfolio_tab(data)` (line 482): No function docstring.
  - `_render_regime_tab(data)` (line 559): No function docstring.
  - `_render_model_tab(data)` (line 653): No function docstring.
  - `_render_features_tab(data)` (line 755): No function docstring.
  - `_render_trades_tab(data)` (line 823): No function docstring.
  - `_render_risk_tab(data)` (line 908): No function docstring.

### `dash_ui/pages/data_explorer.py`
- Lines: 795
- Module intent: Data Explorer -- OHLCV visualization and quality analysis.
- Imports (8): `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `dash.exceptions`, `quant_engine.dash_ui.theme`
- Classes: none
- Top-level functions:
  - `_download_ticker(ticker, timeframe, period)` (line 246): Load OHLCV data for a ticker and timeframe.
  - `_generate_demo_data(ticker)` (line 289): Generate realistic synthetic OHLCV data seeded by ticker name.
  - `_compute_sma(close, window)` (line 316): Simple moving average with NaN fill for insufficient data.
  - `_df_to_store(df)` (line 326): Serialize a DataFrame to a JSON-safe dict for dcc.Store.
  - `_store_to_df(data)` (line 338): Deserialize a dict from dcc.Store back to a DataFrame.
  - `load_data(n_clicks, universe, timeframe, custom_tickers, existing_data)` (line 363): No function docstring.
  - `render_ticker_list(loaded_data, selected_ticker)` (line 418): No function docstring.
  - `select_ticker(n_clicks_list, loaded_data)` (line 471): No function docstring.
  - `update_price_chart(ticker, loaded_data)` (line 489): No function docstring.
  - `update_volume_chart(ticker, loaded_data)` (line 571): No function docstring.
  - `update_stats_bar(ticker, loaded_data)` (line 613): No function docstring.
  - `toggle_quality_modal(open_clicks, close_clicks, is_open, loaded_data)` (line 666): No function docstring.

### `dash_ui/pages/iv_surface.py`
- Lines: 677
- Module intent: IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling.
- Imports (7): `dash`, `dash_bootstrap_components`, `numpy`, `plotly.graph_objects`, `dash`, `quant_engine.dash_ui.components.metric_card`, `quant_engine.dash_ui.theme`
- Classes: none
- Top-level functions:
  - `compute_svi_surface(a, b, rho, m, sigma)` (line 65): Compute SVI implied volatility surface analytically.
  - `compute_svi_smiles(a, b, rho, m, sigma)` (line 83): Compute individual smile curves for select expiries.
  - `_svi_slider(param_id, label, min_val, max_val, step, value, marks)` (line 114): No function docstring.
  - `_build_surface_figure(K_grid, T_grid, iv_grid, title)` (line 300): Create a go.Surface figure with Inferno colorscale.
  - `_build_smiles_figure(smiles, title)` (line 335): Create 2D smile curves for select expiries.
  - `set_svi_preset(preset_name)` (line 369): Set slider values from preset selection.
  - `update_svi_surface(a, b, rho, m, sigma)` (line 388): Recompute SVI surface from slider values (analytical, fast).
  - `set_heston_preset(preset_name)` (line 422): Set Heston slider values from preset.
  - `compute_heston_surface(n_clicks, v0, theta, kappa, sigma, rho)` (line 443): Compute Heston IV surface (computationally intensive).
  - `build_arb_free_surface(n_clicks, spot, rate, div_yield, noise, max_iter)` (line 562): Build arbitrage-free SVI surface from synthetic market data.

### `dash_ui/pages/model_lab.py`
- Lines: 919
- Module intent: Model Lab -- Feature engineering, regime detection, and model training.
- Imports (12): `json`, `traceback`, `pathlib`, `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `dash.exceptions`, `quant_engine.dash_ui.theme`, `quant_engine.dash_ui.data.loaders`
- Classes: none
- Top-level functions:
  - `_demo_feature_importance()` (line 388): Generate plausible demo feature importance data.
  - `load_features(n_clicks)` (line 426): No function docstring.
  - `_demo_regime_payload()` (line 518): Generate plausible demo regime data.
  - `run_regime_detection(n_clicks)` (line 576): No function docstring.
  - `train_model(n_clicks, universe, horizon, feature_mode, options)` (line 741): No function docstring.
  - `_build_cv_chart(cv_scores, holdout_scores)` (line 848): Build a grouped bar chart comparing CV folds and holdout.
  - `_build_demo_cv_chart()` (line 904): Build a demo CV chart with synthetic data.

### `dash_ui/pages/signal_desk.py`
- Lines: 536
- Module intent: Signal Desk -- Prediction generation and signal ranking.
- Imports (9): `pathlib`, `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `dash.exceptions`, `quant_engine.dash_ui.theme`
- Classes: none
- Top-level functions:
  - `_generate_demo_signals(tickers, horizon, top_n)` (line 184): Generate realistic demo signal data for the given tickers.
  - `_try_live_signals(tickers, horizon, top_n)` (line 225): Attempt to generate real signals using the model pipeline.
  - `generate_signals(n_clicks, horizon, top_n)` (line 306): No function docstring.

### `dash_ui/pages/sp_comparison.py`
- Lines: 501
- Module intent: S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation.
- Imports (10): `dash`, `dash_bootstrap_components`, `numpy`, `pandas`, `plotly.graph_objects`, `dash`, `plotly.subplots`, `quant_engine.dash_ui.components.metric_card`, `quant_engine.dash_ui.theme`, `quant_engine.dash_ui.data.loaders`
- Classes: none
- Top-level functions:
  - `load_and_compare(n_clicks, period)` (line 161): Load portfolio and benchmark returns, compute comparison metrics.
  - `toggle_animation(n_clicks, currently_disabled)` (line 433): Toggle the animation interval on/off.
  - `animate_equity_chart(n_intervals, store_data, frame_data)` (line 452): Progressively reveal data points on the main equity chart.

### `dash_ui/pages/system_health.py`
- Lines: 1081
- Module intent: System Health Console -- comprehensive health assessment for the Quant Engine.
- Imports (13): `__future__`, `traceback`, `dataclasses`, `datetime`, `typing`, `dash`, `dash_bootstrap_components`, `numpy`, `plotly.graph_objects`, `dash`, `quant_engine.dash_ui.components.metric_card`, `quant_engine.dash_ui.data.loaders`, `quant_engine.dash_ui.theme`
- Classes: none
- Top-level functions:
  - `_card_panel(title, children, extra_style)` (line 53): Wrap children inside a styled card-panel div.
  - `_status_span(status, text)` (line 68): Return a colored status span with icon.
  - `_check_row(check)` (line 82): Render a single health check as a row.
  - `_instruction_banner(text)` (line 129): Amber instruction banner at top of a tab.
  - `_score_color(score)` (line 146): Return color based on score threshold.
  - `load_health_data(n_intervals, n_clicks)` (line 230): Run collect_health_data() and serialize to JSON-safe dict.
  - `update_health_cards(data)` (line 285): No function docstring.
  - `render_health_tab(tab, data)` (line 344): No function docstring.
  - `_render_overview_tab(data)` (line 375): No function docstring.
  - `_render_data_tab(data)` (line 495): No function docstring.
  - `_render_promotion_tab(data)` (line 584): No function docstring.
  - `_render_wf_tab(data)` (line 692): No function docstring.
  - `_render_execution_tab(data)` (line 830): No function docstring.
  - `_render_complexity_tab(data)` (line 938): No function docstring.

### `dash_ui/server.py`
- Lines: 52
- Module intent: Entry point for the Quant Engine Dash application.
- Imports (5): `tempfile`, `pathlib`, `flask_caching`, `.app`, `.data.cache`
- Classes: none
- Top-level functions: none

### `dash_ui/theme.py`
- Lines: 205
- Module intent: Bloomberg-inspired dark theme for the Quant Engine Dash Dashboard.
- Imports (2): `plotly.graph_objects`, `plotly.io`
- Classes: none
- Top-level functions:
  - `apply_plotly_template()` (line 64): Register and activate the Bloomberg dark Plotly template.
  - `create_figure(kwargs)` (line 108): Create a Plotly figure with the Bloomberg dark template applied.
  - `empty_figure(message)` (line 129): Return an empty Plotly figure with a centered message.
  - `format_pct(value, decimals)` (line 155): Format a decimal value as percentage string.
  - `format_number(value, decimals)` (line 171): Format a number with thousand separators and decimal places.
  - `metric_color(value, positive_is_good)` (line 187): Return color hex code based on value sign and preference.

## Package `data`

### `data/__init__.py`
- Lines: 13
- Module intent: Data subpackage — self-contained data loading, caching, WRDS, and survivorship.
- Imports (5): `.loader`, `.local_cache`, `.provider_registry`, `.quality`, `.feature_store`
- Classes: none
- Top-level functions: none

### `data/alternative.py`
- Lines: 652
- Module intent: Alternative data framework — WRDS-backed implementation.
- Imports (7): `__future__`, `logging`, `datetime`, `pathlib`, `typing`, `numpy`, `pandas`
- Classes:
  - `AlternativeDataProvider` (line 51): WRDS-backed alternative data provider.
    - `__init__(self, cache_dir)` (line 62): Initialise the provider.
    - `_resolve_permno(self, ticker, as_of)` (line 78): Resolve *ticker* to a CRSP PERMNO via the WRDS provider.
    - `get_earnings_surprise(self, ticker, lookback_days)` (line 86): Return earnings surprise data for *ticker* from WRDS I/B/E/S.
    - `get_options_flow(self, ticker)` (line 180): Return options flow data for *ticker* from WRDS OptionMetrics.
    - `get_short_interest(self, ticker)` (line 250): Return short interest data for *ticker* from Compustat via WRDS.
    - `get_insider_transactions(self, ticker)` (line 332): Return insider transaction data for *ticker* from TFN via WRDS.
    - `get_institutional_ownership(self, ticker)` (line 448): Return institutional ownership data with QoQ change features.
- Top-level functions:
  - `_get_wrds()` (line 31): Return the cached WRDSProvider singleton, or None.
  - `compute_alternative_features(ticker, provider, cache_dir)` (line 521): Gather all available alternative data and return as a feature DataFrame.

### `data/feature_store.py`
- Lines: 308
- Module intent: Point-in-time feature store for backtest acceleration.
- Imports (8): `__future__`, `json`, `logging`, `datetime`, `pathlib`, `typing`, `pandas`, `..config`
- Classes:
  - `FeatureStore` (line 38): Point-in-time feature store for backtest acceleration.
    - `__init__(self, store_dir)` (line 49): No method docstring.
    - `_version_dir(self, permno, feature_version)` (line 57): No method docstring.
    - `_ts_tag(computed_at)` (line 61): Normalise *computed_at* to ``YYYY-MM-DD`` for file names.
    - `_parquet_path(self, permno, feature_version, computed_at)` (line 66): No method docstring.
    - `_meta_path(self, permno, feature_version, computed_at)` (line 70): No method docstring.
    - `save_features(self, permno, features, computed_at, feature_version)` (line 78): Persist a feature DataFrame with point-in-time metadata.
    - `load_features(self, permno, as_of, feature_version)` (line 136): Load the most recent feature set for *permno*, respecting
    - `list_available(self, permno)` (line 212): List stored feature snapshots.
    - `invalidate(self, permno, feature_version)` (line 257): Remove cached features for a security.
- Top-level functions: none

### `data/loader.py`
- Lines: 707
- Module intent: Data loader — self-contained data loading with multiple sources.
- Imports (8): `datetime`, `typing`, `numpy`, `pandas`, `..config`, `.local_cache`, `.provider_registry`, `.quality`
- Classes: none
- Top-level functions:
  - `_permno_from_meta(meta)` (line 42): No function docstring.
  - `_ticker_from_meta(meta)` (line 54): No function docstring.
  - `_attach_id_attrs(df, permno, ticker)` (line 61): No function docstring.
  - `_cache_source(meta)` (line 76): No function docstring.
  - `_cache_is_usable(cached, meta, years, require_recent, require_trusted)` (line 81): No function docstring.
  - `_cached_universe_subset(candidates)` (line 117): Prefer locally cached symbols to keep offline runs deterministic.
  - `_normalize_ohlcv(df)` (line 140): Return a sorted, deterministic OHLCV frame or None if invalid.
  - `_harmonize_return_columns(df)` (line 155): Standardize return columns so backtests can consume total-return streams.
  - `_merge_option_surface_from_prefetch(df, permno, option_surface)` (line 191): Merge pre-fetched OptionMetrics surface rows into a single PERMNO panel.
  - `load_ohlcv(ticker, years, use_cache, use_wrds)` (line 222): Load daily OHLCV data for a single ticker.
  - `load_universe(tickers, years, verbose, use_cache, use_wrds)` (line 402): Load OHLCV data for multiple symbols. Returns {permno: DataFrame}.
  - `load_survivorship_universe(as_of_date, years, verbose)` (line 443): Load a survivorship-bias-free universe using WRDS CRSP.
  - `load_with_delistings(tickers, years, verbose)` (line 567): Load OHLCV data including delisting returns from CRSP.

### `data/local_cache.py`
- Lines: 674
- Module intent: Local data cache for daily OHLCV data.
- Imports (6): `json`, `datetime`, `pathlib`, `typing`, `pandas`, `..config`
- Classes: none
- Top-level functions:
  - `_ensure_cache_dir()` (line 25): Create cache directory if it doesn't exist.
  - `_normalize_ohlcv_columns(df)` (line 31): Normalize OHLCV column names to quant_engine's canonical schema.
  - `_to_daily_ohlcv(df)` (line 68): Convert any candidate frame into validated daily OHLCV.
  - `_read_csv_ohlcv(path)` (line 90): No function docstring.
  - `_candidate_csv_paths(cache_root, ticker)` (line 105): No function docstring.
  - `_cache_meta_path(data_path, ticker)` (line 121): No function docstring.
  - `_read_cache_meta(data_path, ticker)` (line 134): No function docstring.
  - `_write_cache_meta(data_path, ticker, df, source, meta)` (line 152): No function docstring.
  - `save_ohlcv(ticker, df, cache_dir, source, meta)` (line 178): Save OHLCV DataFrame to local cache.
  - `load_ohlcv_with_meta(ticker, cache_dir)` (line 213): Load OHLCV and sidecar metadata from cache roots.
  - `load_ohlcv(ticker, cache_dir)` (line 287): Load OHLCV DataFrame from local cache.
  - `load_intraday_ohlcv(ticker, timeframe, cache_dir)` (line 303): Load intraday OHLCV data from cache.
  - `list_intraday_timeframes(ticker, cache_dir)` (line 370): Return list of available intraday timeframes for a ticker in the cache.
  - `list_cached_tickers(cache_dir)` (line 384): List all tickers available in cache roots.
  - `_daily_cache_files(root)` (line 403): Return de-duplicated daily-cache candidate files for one root.
  - `_ticker_from_cache_path(path)` (line 426): No function docstring.
  - `_timeframe_from_cache_path(path)` (line 440): Determine the canonical timeframe from a cache file path.
  - `_all_cache_files(root)` (line 451): Return de-duplicated daily + intraday cache candidate files for one root.
  - `rehydrate_cache_metadata(cache_roots, source_by_root, default_source, only_missing, overwrite_source, dry_run)` (line 473): Backfill metadata sidecars for existing cache files without rewriting price data.
  - `load_ibkr_data(data_dir)` (line 613): Scan a directory of IBKR-downloaded files (CSV or parquet).
  - `cache_universe(data, cache_dir, source)` (line 667): Save all tickers in a data dict to the local cache.

### `data/provider_base.py`
- Lines: 12
- Module intent: Shared provider protocol for pluggable data connectors.
- Imports (2): `__future__`, `typing`
- Classes:
  - `DataProvider` (line 9): No class docstring.
    - `available(self)` (line 10): No method docstring.
- Top-level functions: none

### `data/provider_registry.py`
- Lines: 49
- Module intent: Provider registry for unified data-provider access (WRDS, Kalshi, ...).
- Imports (3): `__future__`, `typing`, `.provider_base`
- Classes: none
- Top-level functions:
  - `_wrds_factory(kwargs)` (line 14): No function docstring.
  - `_kalshi_factory(kwargs)` (line 20): No function docstring.
  - `get_provider(name, kwargs)` (line 32): No function docstring.
  - `list_providers()` (line 40): No function docstring.
  - `register_provider(name, factory)` (line 44): No function docstring.

### `data/quality.py`
- Lines: 237
- Module intent: Data quality checks for OHLCV time series.
- Imports (5): `dataclasses`, `typing`, `numpy`, `pandas`, `..config`
- Classes:
  - `DataQualityReport` (line 22): No class docstring.
    - `to_dict(self)` (line 27): No method docstring.
- Top-level functions:
  - `assess_ohlcv_quality(df, max_missing_bar_fraction, max_zero_volume_fraction, max_abs_daily_return)` (line 31): No function docstring.
  - `generate_quality_report(ohlcv_dict, quality_weights, degraded_threshold, extreme_return_pct)` (line 96): Return a per-stock quality summary DataFrame.
  - `flag_degraded_stocks(ohlcv_dict, degraded_threshold, extreme_return_pct)` (line 204): Return a list of tickers whose data quality is below threshold.

### `data/survivorship.py`
- Lines: 928
- Module intent: Survivorship Bias Controls (Tasks 112-117)
- Imports (9): `numpy`, `pandas`, `dataclasses`, `typing`, `datetime`, `enum`, `json`, `sqlite3`, `os`
- Classes:
  - `DelistingReason` (line 25): Reason for stock delisting.
    - Methods: none
  - `UniverseMember` (line 38): Task 112: Track a symbol's membership in a universe.
    - `is_active_on(self, check_date)` (line 50): Check if symbol was in universe on given date.
    - `to_dict(self)` (line 58): No method docstring.
  - `UniverseChange` (line 71): Task 114: Track a change to universe membership.
    - `to_dict(self)` (line 83): No method docstring.
  - `DelistingEvent` (line 96): Task 113: Track delisting event with proper returns.
    - `to_dict(self)` (line 110): No method docstring.
  - `SurvivorshipReport` (line 125): Task 117: Report comparing returns with/without survivorship adjustment.
    - `to_dict(self)` (line 149): No method docstring.
  - `UniverseHistoryTracker` (line 167): Task 112, 114, 115: Track historical universe membership.
    - `__init__(self, db_path)` (line 174): No method docstring.
    - `_init_db(self)` (line 178): Initialize database schema.
    - `add_member(self, member)` (line 220): Add a universe member record.
    - `record_change(self, change)` (line 239): Task 114: Record a universe change.
    - `get_universe_on_date(self, universe_name, as_of_date)` (line 258): Task 115: Reconstruct universe membership on a specific date.
    - `get_changes_in_period(self, universe_name, start_date, end_date)` (line 282): Get all changes to a universe in a period.
    - `bulk_load_universe(self, universe_name, symbols, as_of_date)` (line 318): Bulk load current universe members.
    - `clear_universe(self, universe_name)` (line 334): Remove stored membership records for one universe.
  - `DelistingHandler` (line 550): Task 113, 116: Handle delisting events properly.
    - `__init__(self, db_path)` (line 560): No method docstring.
    - `_init_db(self)` (line 564): Initialize database schema.
    - `record_delisting(self, event)` (line 603): Task 113: Record a delisting event.
    - `preserve_price_history(self, symbol, prices)` (line 624): Task 116: Preserve price history for dead company.
    - `get_dead_company_prices(self, symbol)` (line 652): Task 116: Retrieve preserved price history for dead company.
    - `get_delisting_event(self, symbol)` (line 670): Get delisting event for a symbol.
    - `get_delisting_return(self, symbol)` (line 695): Task 113: Get delisting return for proper backtest accounting.
    - `is_delisted(self, symbol, as_of_date)` (line 706): Check if symbol was delisted by a given date.
    - `get_all_delisted_symbols(self)` (line 713): Get list of all delisted symbols.
  - `SurvivorshipBiasController` (line 722): Task 117: Main controller for survivorship bias analysis.
    - `__init__(self, universe_tracker, delisting_handler)` (line 732): No method docstring.
    - `get_survivorship_free_universe(self, universe_name, as_of_date)` (line 740): Get universe membership as it would have been known on a date.
    - `calculate_bias_impact(self, universe_name, prices, start_date, end_date, current_survivors_only)` (line 755): Task 117: Calculate the impact of survivorship bias.
    - `format_report(self, report)` (line 863): Format survivorship bias report.
- Top-level functions:
  - `hydrate_universe_history_from_snapshots(snapshots, universe_name, db_path, verbose)` (line 348): Build point-in-time universe intervals from snapshot rows.
  - `hydrate_sp500_history_from_wrds(start_date, end_date, db_path, freq, verbose)` (line 445): Pull historical S&P 500 snapshots from WRDS and hydrate local PIT DB.
  - `filter_panel_by_point_in_time_universe(panel, universe_name, db_path, verbose)` (line 477): Filter MultiIndex panel rows by point-in-time universe membership.
  - `reconstruct_historical_universe(universe_name, as_of_date, tracker)` (line 901): Task 115: Quick function to reconstruct historical universe.
  - `calculate_survivorship_bias_impact(prices, start_date, end_date, universe_name)` (line 913): Task 117: Quick function to calculate survivorship bias impact.

### `data/wrds_provider.py`
- Lines: 1584
- Module intent: wrds_provider.py
- Imports (7): `os`, `warnings`, `datetime`, `typing`, `numpy`, `pandas`, `re`
- Classes:
  - `WRDSProvider` (line 198): WRDS data provider for the auto-discovery pipeline.
    - `__init__(self)` (line 206): No method docstring.
    - `available(self)` (line 211): No method docstring.
    - `_query(self, sql)` (line 214): Run a SQL query and return a DataFrame. Returns empty df on error.
    - `_query_silent(self, sql)` (line 224): Run a SQL query but suppress probing errors (used for optional tables).
    - `get_sp500_universe(self, as_of_date, include_permno)` (line 237): Return the list of tickers that were IN the S&P 500 on a specific date.
    - `get_sp500_history(self, start_date, end_date, freq)` (line 332): Return the full history of S&P 500 constituents over a date range.
    - `resolve_permno(self, ticker, as_of_date)` (line 377): Resolve a ticker to the active CRSP PERMNO on a date.
    - `get_crsp_prices(self, tickers, start_date, end_date)` (line 406): Fetch daily OHLCV-equivalent data from CRSP.
    - `get_crsp_prices_with_delistings(self, tickers, start_date, end_date)` (line 500): Same as get_crsp_prices but integrates delisting returns into
    - `get_optionmetrics_link(self, permnos, start_date, end_date)` (line 580): Resolve WRDS OptionMetrics secid link rows for PERMNOs.
    - `_nearest_iv(group, target_dte, cp_flag, delta_target)` (line 629): No method docstring.
    - `get_option_surface_features(self, permnos, start_date, end_date)` (line 651): Build minimal daily option-surface features keyed by (permno, date).
    - `get_fundamentals(self, tickers, start_date, end_date)` (line 804): Fetch quarterly fundamentals from Compustat with point-in-time dates.
    - `get_earnings_surprises(self, tickers, start_date, end_date)` (line 887): Fetch quarterly earnings announcements and EPS surprises from I/B/E/S.
    - `get_institutional_ownership(self, tickers, start_date, end_date)` (line 957): Fetch quarterly institutional ownership from 13F filings (TFN s34).
    - `get_taqmsec_ohlcv(self, ticker, timeframe, start_date, end_date)` (line 1017): Build OHLCV bars from WRDS TAQmsec tick data for a single ticker.
    - `query_options_volume(self, permno, start_date, end_date)` (line 1152): Query OptionMetrics ``optionm.opprcd`` for daily aggregate options
    - `query_short_interest(self, permno, start_date, end_date)` (line 1239): Query Compustat ``comp.sec_shortint`` (or equivalent table) for
    - `query_insider_transactions(self, permno, start_date, end_date)` (line 1396): Query insider transaction data from TFN tables for Form 4 filings.
    - `_permno_to_ticker(self, permno, as_of_date)` (line 1510): Resolve a PERMNO to its most recent ticker symbol.
    - `get_ohlcv(self, ticker, start_date, end_date)` (line 1540): Single-ticker OHLCV fetch from CRSP (drop-in for DataLayer.get_ohlcv).
- Top-level functions:
  - `_sanitize_ticker_list(tickers)` (line 67): Build a SQL-safe IN-clause string from ticker symbols.
  - `_sanitize_permno_list(permnos)` (line 83): Build a SQL-safe IN-clause string from PERMNO values.
  - `_read_pgpass_password()` (line 98): Read the WRDS password from ~/.pgpass so the wrds library doesn't
  - `_get_connection()` (line 132): Get or create a cached WRDS connection. Returns None if unavailable.
  - `get_wrds_provider()` (line 1571): Get or create the default WRDSProvider singleton.
  - `wrds_available()` (line 1579): Quick check: is WRDS accessible?

## Package `features`

### `features/__init__.py`
- Lines: 0
- Module intent: No module docstring.
- Imports: none
- Classes: none
- Top-level functions: none

### `features/harx_spillovers.py`
- Lines: 242
- Module intent: HARX Volatility Spillover features (Tier 6.1).
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `_realized_volatility(returns, window, min_periods)` (line 38): Rolling realized volatility (annualised std of returns).
  - `_ols_lstsq(X, y)` (line 48): OLS via numpy lstsq.  Returns coefficient vector.
  - `compute_harx_spillovers(returns_by_asset, rv_daily_window, rv_weekly_window, rv_monthly_window, regression_window, min_regression_obs)` (line 54): Compute HARX cross-market volatility spillover features.

### `features/intraday.py`
- Lines: 193
- Module intent: Intraday microstructure features from WRDS TAQmsec tick data.
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `compute_intraday_features(ticker, date, wrds_provider)` (line 22): Compute intraday microstructure features for a single ticker on a date.

### `features/lob_features.py`
- Lines: 311
- Module intent: Markov LOB (Limit Order Book) features from intraday bar data (Tier 6.2).
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `_inter_bar_durations(index)` (line 34): Compute inter-bar durations in seconds from a DatetimeIndex.
  - `_estimate_poisson_lambda(durations)` (line 51): Estimate trade arrival rate (lambda) from inter-arrival durations.
  - `_signed_volume(bars)` (line 72): Approximate trade direction using candle body (close - open).
  - `compute_lob_features(intraday_bars, freq)` (line 88): Compute Markov LOB proxy features for a single stock-day.
  - `compute_lob_features_batch(intraday_data, freq)` (line 239): Compute LOB features for multiple stock-days in batch.

### `features/macro.py`
- Lines: 243
- Module intent: FRED macro indicator features for quant_engine.
- Imports (7): `__future__`, `hashlib`, `logging`, `pathlib`, `typing`, `numpy`, `pandas`
- Classes:
  - `MacroFeatureProvider` (line 50): FRED API integration for macro indicator features.
    - `__init__(self, api_key, cache_dir)` (line 63): No method docstring.
    - `_fetch_series_fredapi(self, series_id, start_date, end_date)` (line 81): Fetch a FRED series using the ``fredapi`` library.
    - `_fetch_series_requests(self, series_id, start_date, end_date)` (line 100): Fetch a FRED series using raw ``requests`` (fallback).
    - `_fetch_series(self, series_id, start_date, end_date)` (line 140): Fetch a single FRED series with library fallback and caching.
    - `get_macro_features(self, start_date, end_date)` (line 195): Download FRED macro series and compute level + momentum features.
- Top-level functions:
  - `_cache_key(series_id, start, end)` (line 43): Generate a deterministic cache filename.

### `features/options_factors.py`
- Lines: 119
- Module intent: Option surface factor construction from OptionMetrics-enriched daily panels.
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `_pick_numeric(df, candidates)` (line 12): No function docstring.
  - `_rolling_percentile_rank(series, window, min_periods)` (line 19): No function docstring.
  - `compute_option_surface_factors(df)` (line 36): Compute minimal high-signal option surface features.
  - `compute_iv_shock_features(df, window)` (line 84): Event-centric IV shock features (G3).

### `features/pipeline.py`
- Lines: 819
- Module intent: Feature Pipeline — computes model features from OHLCV data.
- Imports (7): `typing`, `numpy`, `pandas`, `..indicators`, `.research_factors`, `.options_factors`, `.wave_flow`
- Classes:
  - `FeaturePipeline` (line 427): End-to-end feature computation pipeline.
    - `__init__(self, feature_mode, include_interactions, include_research_factors, include_cross_asset_factors, include_options_factors, research_config, verbose)` (line 436): No method docstring.
    - `compute(self, df, compute_targets_flag, benchmark_close)` (line 462): Compute all features and targets from OHLCV data.
    - `compute_universe(self, data, verbose, compute_targets_flag, benchmark_close)` (line 554): Compute features for all symbols, stacking into a single DataFrame.
    - `_load_benchmark_close(verbose)` (line 803): Load benchmark close prices for excess-return target computation.
- Top-level functions:
  - `_build_indicator_set()` (line 68): Instantiate all indicators with default parameters.
  - `_get_indicators()` (line 153): No function docstring.
  - `compute_indicator_features(df, verbose)` (line 160): Compute all indicator-based features as continuous columns.
  - `compute_raw_features(df)` (line 186): Compute raw OHLCV-derived features (returns, volume, gaps, etc.).
  - `compute_har_volatility_features(df)` (line 228): Compute HAR (Heterogeneous Autoregressive) realized volatility features.
  - `compute_multiscale_features(df)` (line 267): Compute momentum, RSI, and volatility features at multiple time scales.
  - `compute_interaction_features(features, pairs)` (line 303): Generate interaction features from pairs of continuous indicators.
  - `compute_targets(df, horizons, benchmark_close)` (line 356): Compute forward return targets for supervised learning.
  - `_winsorize_expanding(df, lower_q, upper_q)` (line 405): Winsorize features using expanding-window quantiles (no look-ahead).

### `features/research_factors.py`
- Lines: 976
- Module intent: Research-derived factor construction for quant_engine.
- Imports (5): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `ResearchFactorConfig` (line 28): Configuration for research-derived factor generation.
    - Methods: none
- Top-level functions:
  - `_rolling_zscore(series, window, min_periods)` (line 43): Causal rolling z-score.
  - `_safe_pct_change(series, periods)` (line 50): No function docstring.
  - `_required_ohlcv(df)` (line 55): No function docstring.
  - `compute_order_flow_impact_factors(df, config)` (line 64): Order-flow imbalance and price-impact proxies (Cont et al. inspired).
  - `compute_markov_queue_features(df, config)` (line 132): Markov-style queue imbalance features (de Larrard style state framing).
  - `compute_time_series_momentum_factors(df, config)` (line 202): Vol-scaled time-series momentum factors (Moskowitz/Ooi/Pedersen style).
  - `compute_vol_scaled_momentum(df, horizons, vol_window)` (line 243): Volatility-scaled time-series momentum enhancements.
  - `_rolling_levy_area(dx, dy, window, min_periods)` (line 335): Rolling Levy area for a 2D path of increments.
  - `compute_signature_path_features(df, config)` (line 361): Signature-inspired path features for returns-volume trajectory.
  - `compute_vol_surface_factors(df, config)` (line 396): Volatility term-structure factors inspired by implied-vol surface dynamics.
  - `compute_single_asset_research_factors(df, config)` (line 465): Compute all single-asset research factors.
  - `_standardize_block(block)` (line 483): Column-wise z-score with NaN-safe handling.
  - `_lagged_weight_matrix(values, t, window, min_obs)` (line 492): Build positive lagged correlation weights:
  - `compute_cross_asset_research_factors(price_data, config)` (line 531): Compute cross-asset network momentum and volatility spillover factors.
  - `_dtw_distance_numpy(x, y)` (line 643): Pure numpy DTW distance computation using dynamic programming.
  - `_dtw_avg_lag_from_path(path)` (line 683): Extract average lag from DTW alignment path.
  - `compute_dtw_lead_lag(returns, window, max_lag)` (line 694): DTW-based lead-lag detection across a universe of assets.
  - `_numpy_order2_signature(price_inc, volume_inc)` (line 849): Pure numpy computation of truncated order-2 path signature for a 2D path
  - `compute_path_signatures(df, windows, order)` (line 892): Compute truncated path signatures of (price, volume) paths.

### `features/wave_flow.py`
- Lines: 144
- Module intent: Wave-Flow Decomposition for quant_engine.
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `compute_wave_flow_decomposition(df, short_window, long_window, regime_threshold)` (line 26): Decompose the return series into flow (secular trend) and wave (oscillatory)

## Package `indicators`

### `indicators/__init__.py`
- Lines: 57
- Module intent: Quant Engine Indicators — self-contained copy of the technical indicator library.
- Imports (1): `.indicators`
- Classes: none
- Top-level functions: none

### `indicators/indicators.py`
- Lines: 2635
- Module intent: Technical Indicator Library
- Imports (4): `numpy`, `pandas`, `typing`, `abc`
- Classes:
  - `Indicator` (line 14): Base class for all indicators.
    - `name(self)` (line 19): No method docstring.
    - `calculate(self, df)` (line 23): Calculate indicator values. Returns a Series.
  - `ATR` (line 32): Average True Range - measures volatility.
    - `__init__(self, period)` (line 35): No method docstring.
    - `name(self)` (line 39): No method docstring.
    - `calculate(self, df)` (line 42): No method docstring.
  - `NATR` (line 56): Normalized ATR - ATR as percentage of close price.
    - `__init__(self, period)` (line 59): No method docstring.
    - `name(self)` (line 64): No method docstring.
    - `calculate(self, df)` (line 67): No method docstring.
  - `BollingerBandWidth` (line 73): Bollinger Band Width - measures volatility squeeze.
    - `__init__(self, period, std_dev)` (line 76): No method docstring.
    - `name(self)` (line 81): No method docstring.
    - `calculate(self, df)` (line 84): No method docstring.
  - `HistoricalVolatility` (line 96): Historical volatility (standard deviation of returns).
    - `__init__(self, period)` (line 99): No method docstring.
    - `name(self)` (line 103): No method docstring.
    - `calculate(self, df)` (line 106): No method docstring.
  - `RSI` (line 116): Relative Strength Index.
    - `__init__(self, period)` (line 119): No method docstring.
    - `name(self)` (line 123): No method docstring.
    - `calculate(self, df)` (line 126): No method docstring.
  - `MACD` (line 140): MACD Line (difference between fast and slow EMA).
    - `__init__(self, fast, slow)` (line 143): No method docstring.
    - `name(self)` (line 148): No method docstring.
    - `calculate(self, df)` (line 151): No method docstring.
  - `MACDSignal` (line 158): MACD Signal Line.
    - `__init__(self, fast, slow, signal)` (line 161): No method docstring.
    - `name(self)` (line 168): No method docstring.
    - `calculate(self, df)` (line 171): No method docstring.
  - `MACDHistogram` (line 177): MACD Histogram (MACD - Signal).
    - `__init__(self, fast, slow, signal)` (line 180): No method docstring.
    - `name(self)` (line 188): No method docstring.
    - `calculate(self, df)` (line 191): No method docstring.
  - `ROC` (line 197): Rate of Change.
    - `__init__(self, period)` (line 200): No method docstring.
    - `name(self)` (line 204): No method docstring.
    - `calculate(self, df)` (line 207): No method docstring.
  - `Stochastic` (line 213): Stochastic %K.
    - `__init__(self, period)` (line 216): No method docstring.
    - `name(self)` (line 220): No method docstring.
    - `calculate(self, df)` (line 223): No method docstring.
  - `StochasticD` (line 231): Stochastic %D (smoothed %K).
    - `__init__(self, k_period, d_period)` (line 234): No method docstring.
    - `name(self)` (line 240): No method docstring.
    - `calculate(self, df)` (line 243): No method docstring.
  - `WilliamsR` (line 249): Williams %R.
    - `__init__(self, period)` (line 252): No method docstring.
    - `name(self)` (line 256): No method docstring.
    - `calculate(self, df)` (line 259): No method docstring.
  - `CCI` (line 267): Commodity Channel Index.
    - `__init__(self, period)` (line 270): No method docstring.
    - `name(self)` (line 274): No method docstring.
    - `calculate(self, df)` (line 277): No method docstring.
  - `SMA` (line 291): Simple Moving Average.
    - `__init__(self, period)` (line 294): No method docstring.
    - `name(self)` (line 298): No method docstring.
    - `calculate(self, df)` (line 301): No method docstring.
  - `EMA` (line 305): Exponential Moving Average.
    - `__init__(self, period)` (line 308): No method docstring.
    - `name(self)` (line 312): No method docstring.
    - `calculate(self, df)` (line 315): No method docstring.
  - `PriceVsSMA` (line 319): Price distance from SMA (as percentage).
    - `__init__(self, period)` (line 322): No method docstring.
    - `name(self)` (line 327): No method docstring.
    - `calculate(self, df)` (line 330): No method docstring.
  - `SMASlope` (line 335): Slope of SMA (rate of change).
    - `__init__(self, sma_period, slope_period)` (line 338): No method docstring.
    - `name(self)` (line 344): No method docstring.
    - `calculate(self, df)` (line 347): No method docstring.
  - `ADX` (line 353): Average Directional Index - trend strength.
    - `__init__(self, period)` (line 356): No method docstring.
    - `name(self)` (line 360): No method docstring.
    - `calculate(self, df)` (line 363): No method docstring.
  - `Aroon` (line 388): Aroon Oscillator.
    - `__init__(self, period)` (line 391): No method docstring.
    - `name(self)` (line 395): No method docstring.
    - `calculate(self, df)` (line 398): No method docstring.
  - `VolumeRatio` (line 417): Current volume vs average volume.
    - `__init__(self, period)` (line 420): No method docstring.
    - `name(self)` (line 424): No method docstring.
    - `calculate(self, df)` (line 427): No method docstring.
  - `OBV` (line 432): On-Balance Volume.
    - `name(self)` (line 436): No method docstring.
    - `calculate(self, df)` (line 439): No method docstring.
  - `OBVSlope` (line 450): OBV rate of change.
    - `__init__(self, period)` (line 453): No method docstring.
    - `name(self)` (line 458): No method docstring.
    - `calculate(self, df)` (line 461): No method docstring.
  - `MFI` (line 468): Money Flow Index.
    - `__init__(self, period)` (line 471): No method docstring.
    - `name(self)` (line 475): No method docstring.
    - `calculate(self, df)` (line 478): No method docstring.
  - `HigherHighs` (line 496): Count of higher highs in lookback period.
    - `__init__(self, period)` (line 499): No method docstring.
    - `name(self)` (line 503): No method docstring.
    - `calculate(self, df)` (line 506): No method docstring.
  - `LowerLows` (line 512): Count of lower lows in lookback period.
    - `__init__(self, period)` (line 515): No method docstring.
    - `name(self)` (line 519): No method docstring.
    - `calculate(self, df)` (line 522): No method docstring.
  - `CandleBody` (line 528): Candle body size as percentage of range.
    - `name(self)` (line 532): No method docstring.
    - `calculate(self, df)` (line 535): No method docstring.
  - `CandleDirection` (line 541): Candle direction streak (positive = up candles, negative = down).
    - `__init__(self, period)` (line 544): No method docstring.
    - `name(self)` (line 548): No method docstring.
    - `calculate(self, df)` (line 551): No method docstring.
  - `GapPercent` (line 556): Gap from previous close as percentage.
    - `name(self)` (line 560): No method docstring.
    - `calculate(self, df)` (line 563): No method docstring.
  - `DistanceFromHigh` (line 571): Distance from N-period high as percentage.
    - `__init__(self, period)` (line 574): No method docstring.
    - `name(self)` (line 578): No method docstring.
    - `calculate(self, df)` (line 581): No method docstring.
  - `DistanceFromLow` (line 586): Distance from N-period low as percentage.
    - `__init__(self, period)` (line 589): No method docstring.
    - `name(self)` (line 593): No method docstring.
    - `calculate(self, df)` (line 596): No method docstring.
  - `PricePercentile` (line 601): Current price percentile within N-period range.
    - `__init__(self, period)` (line 604): No method docstring.
    - `name(self)` (line 608): No method docstring.
    - `calculate(self, df)` (line 611): No method docstring.
  - `BBWidthPercentile` (line 622): Bollinger Band Width Percentile - identifies squeeze conditions.
    - `__init__(self, bb_period, lookback)` (line 628): No method docstring.
    - `name(self)` (line 634): No method docstring.
    - `calculate(self, df)` (line 637): No method docstring.
  - `NATRPercentile` (line 646): NATR Percentile - where current volatility sits vs history.
    - `__init__(self, natr_period, lookback)` (line 652): No method docstring.
    - `name(self)` (line 658): No method docstring.
    - `calculate(self, df)` (line 661): No method docstring.
  - `VolatilitySqueeze` (line 670): Volatility Squeeze indicator - BB inside Keltner Channel.
    - `__init__(self, period, bb_mult, kc_mult)` (line 676): No method docstring.
    - `name(self)` (line 682): No method docstring.
    - `calculate(self, df)` (line 685): No method docstring.
  - `RVOL` (line 708): Relative Volume - current volume vs same time period average.
    - `__init__(self, period)` (line 714): No method docstring.
    - `name(self)` (line 718): No method docstring.
    - `calculate(self, df)` (line 721): No method docstring.
  - `NetVolumeTrend` (line 727): Net Volume Trend - accumulation/distribution pressure.
    - `__init__(self, period)` (line 733): No method docstring.
    - `name(self)` (line 737): No method docstring.
    - `calculate(self, df)` (line 740): No method docstring.
  - `VolumeForce` (line 755): Volume Force Index - measures buying/selling pressure.
    - `__init__(self, period)` (line 761): No method docstring.
    - `name(self)` (line 765): No method docstring.
    - `calculate(self, df)` (line 768): No method docstring.
  - `AccumulationDistribution` (line 774): Accumulation/Distribution Line slope.
    - `__init__(self, period)` (line 780): No method docstring.
    - `name(self)` (line 784): No method docstring.
    - `calculate(self, df)` (line 787): No method docstring.
  - `EMAAlignment` (line 799): EMA Alignment - checks if EMAs are properly stacked.
    - `__init__(self, fast, medium, slow)` (line 805): No method docstring.
    - `name(self)` (line 811): No method docstring.
    - `calculate(self, df)` (line 814): No method docstring.
  - `TrendStrength` (line 829): Combined trend strength using multiple factors.
    - `__init__(self, period)` (line 835): No method docstring.
    - `name(self)` (line 839): No method docstring.
    - `calculate(self, df)` (line 842): No method docstring.
  - `PriceVsEMAStack` (line 862): Price position relative to EMA stack.
    - `name(self)` (line 869): No method docstring.
    - `calculate(self, df)` (line 872): No method docstring.
  - `PivotHigh` (line 888): Pivot High breakout - price breaks above N-bar high.
    - `__init__(self, left_bars, right_bars)` (line 894): No method docstring.
    - `name(self)` (line 899): No method docstring.
    - `calculate(self, df)` (line 902): No method docstring.
  - `PivotLow` (line 921): Pivot Low breakdown - price breaks below N-bar low.
    - `__init__(self, left_bars, right_bars)` (line 927): No method docstring.
    - `name(self)` (line 932): No method docstring.
    - `calculate(self, df)` (line 935): No method docstring.
  - `NBarHighBreak` (line 954): Simple N-bar high breakout.
    - `__init__(self, period)` (line 960): No method docstring.
    - `name(self)` (line 964): No method docstring.
    - `calculate(self, df)` (line 967): No method docstring.
  - `NBarLowBreak` (line 973): Simple N-bar low breakdown.
    - `__init__(self, period)` (line 979): No method docstring.
    - `name(self)` (line 983): No method docstring.
    - `calculate(self, df)` (line 986): No method docstring.
  - `RangeBreakout` (line 992): Range Breakout - price breaks out of N-day range.
    - `__init__(self, period)` (line 998): No method docstring.
    - `name(self)` (line 1002): No method docstring.
    - `calculate(self, df)` (line 1005): No method docstring.
  - `ATRTrailingStop` (line 1019): Distance from ATR trailing stop.
    - `__init__(self, period, multiplier)` (line 1026): No method docstring.
    - `name(self)` (line 1032): No method docstring.
    - `calculate(self, df)` (line 1035): No method docstring.
  - `ATRChannel` (line 1048): Position within ATR channel.
    - `__init__(self, period, multiplier)` (line 1054): No method docstring.
    - `name(self)` (line 1060): No method docstring.
    - `calculate(self, df)` (line 1063): No method docstring.
  - `RiskPerATR` (line 1075): Recent price range in ATR units.
    - `__init__(self, atr_period, range_period)` (line 1081): No method docstring.
    - `name(self)` (line 1087): No method docstring.
    - `calculate(self, df)` (line 1090): No method docstring.
  - `MarketRegime` (line 1104): Market regime based on price action.
    - `__init__(self, period)` (line 1111): No method docstring.
    - `name(self)` (line 1115): No method docstring.
    - `calculate(self, df)` (line 1118): No method docstring.
  - `VolatilityRegime` (line 1132): Volatility regime classification.
    - `__init__(self, period, lookback)` (line 1138): No method docstring.
    - `name(self)` (line 1143): No method docstring.
    - `calculate(self, df)` (line 1146): No method docstring.
  - `VWAP` (line 1164): Volume Weighted Average Price - rolling calculation.
    - `__init__(self, period)` (line 1170): No method docstring.
    - `name(self)` (line 1174): No method docstring.
    - `calculate(self, df)` (line 1177): No method docstring.
  - `PriceVsVWAP` (line 1184): Price distance from VWAP as percentage.
    - `__init__(self, period)` (line 1190): No method docstring.
    - `name(self)` (line 1195): No method docstring.
    - `calculate(self, df)` (line 1198): No method docstring.
  - `VWAPBands` (line 1203): VWAP Standard Deviation Bands.
    - `__init__(self, period, num_std)` (line 1209): No method docstring.
    - `name(self)` (line 1215): No method docstring.
    - `calculate(self, df)` (line 1218): No method docstring.
  - `AnchoredVWAP` (line 1235): Anchored VWAP - VWAP calculated from N days ago.
    - `__init__(self, anchor_days)` (line 1242): No method docstring.
    - `name(self)` (line 1246): No method docstring.
    - `calculate(self, df)` (line 1249): No method docstring.
  - `PriceVsAnchoredVWAP` (line 1261): Price distance from Anchored VWAP.
    - `__init__(self, anchor_days)` (line 1267): No method docstring.
    - `name(self)` (line 1272): No method docstring.
    - `calculate(self, df)` (line 1275): No method docstring.
  - `MultiVWAPPosition` (line 1280): Position relative to multiple VWAP anchors.
    - `name(self)` (line 1288): No method docstring.
    - `calculate(self, df)` (line 1291): No method docstring.
  - `ValueAreaHigh` (line 1307): Value Area High approximation.
    - `__init__(self, period)` (line 1314): No method docstring.
    - `name(self)` (line 1318): No method docstring.
    - `calculate(self, df)` (line 1321): No method docstring.
  - `ValueAreaLow` (line 1348): Value Area Low approximation.
    - `__init__(self, period)` (line 1354): No method docstring.
    - `name(self)` (line 1358): No method docstring.
    - `calculate(self, df)` (line 1361): No method docstring.
  - `POC` (line 1388): Point of Control approximation.
    - `__init__(self, period, num_bins)` (line 1394): No method docstring.
    - `name(self)` (line 1399): No method docstring.
    - `calculate(self, df)` (line 1402): No method docstring.
  - `PriceVsPOC` (line 1437): Price distance from Point of Control.
    - `__init__(self, period)` (line 1443): No method docstring.
    - `name(self)` (line 1448): No method docstring.
    - `calculate(self, df)` (line 1451): No method docstring.
  - `ValueAreaPosition` (line 1456): Position within Value Area.
    - `__init__(self, period)` (line 1463): No method docstring.
    - `name(self)` (line 1469): No method docstring.
    - `calculate(self, df)` (line 1472): No method docstring.
  - `AboveValueArea` (line 1484): Binary: 1 if price above VAH, 0 otherwise.
    - `__init__(self, period)` (line 1490): No method docstring.
    - `name(self)` (line 1495): No method docstring.
    - `calculate(self, df)` (line 1498): No method docstring.
  - `BelowValueArea` (line 1503): Binary: 1 if price below VAL, 0 otherwise.
    - `__init__(self, period)` (line 1509): No method docstring.
    - `name(self)` (line 1514): No method docstring.
    - `calculate(self, df)` (line 1517): No method docstring.
  - `Beast666Proximity` (line 1526): Beast 666 Proximity Score (0-100).
    - `__init__(self, tolerance)` (line 1578): No method docstring.
    - `name(self)` (line 1589): No method docstring.
    - `calculate(self, df)` (line 1593): No method docstring.
  - `Beast666Distance` (line 1620): Signed percent distance from the nearest 666 level.
    - `__init__(self)` (line 1632): No method docstring.
    - `name(self)` (line 1636): No method docstring.
    - `calculate(self, df)` (line 1639): No method docstring.
  - `ParkinsonVolatility` (line 1670): Parkinson range-based volatility estimator. More efficient than close-to-close.
    - `__init__(self, period)` (line 1673): No method docstring.
    - `name(self)` (line 1677): No method docstring.
    - `calculate(self, df)` (line 1680): No method docstring.
  - `GarmanKlassVolatility` (line 1687): Garman-Klass OHLC volatility estimator. ~8x more efficient than close-to-close.
    - `__init__(self, period)` (line 1690): No method docstring.
    - `name(self)` (line 1694): No method docstring.
    - `calculate(self, df)` (line 1697): No method docstring.
  - `YangZhangVolatility` (line 1705): Yang-Zhang volatility combining overnight and Rogers-Satchell intraday components.
    - `__init__(self, period)` (line 1708): No method docstring.
    - `name(self)` (line 1712): No method docstring.
    - `calculate(self, df)` (line 1715): No method docstring.
  - `VolatilityCone` (line 1736): Percentile rank of current realized vol vs its historical distribution.
    - `__init__(self, vol_period, lookback)` (line 1739): No method docstring.
    - `name(self)` (line 1744): No method docstring.
    - `calculate(self, df)` (line 1747): No method docstring.
  - `VolOfVol` (line 1757): Volatility of volatility - rolling std of rolling volatility.
    - `__init__(self, vol_period, vov_period)` (line 1760): No method docstring.
    - `name(self)` (line 1765): No method docstring.
    - `calculate(self, df)` (line 1768): No method docstring.
  - `GARCHVolatility` (line 1774): Simplified GARCH(1,1) volatility with fixed parameters.
    - `__init__(self, period, alpha, beta)` (line 1777): No method docstring.
    - `name(self)` (line 1784): No method docstring.
    - `calculate(self, df)` (line 1787): No method docstring.
  - `VolTermStructure` (line 1809): Ratio of short-term to long-term realized vol. >1 = backwardation (fear).
    - `__init__(self, short_period, long_period)` (line 1812): No method docstring.
    - `name(self)` (line 1817): No method docstring.
    - `calculate(self, df)` (line 1820): No method docstring.
  - `HurstExponent` (line 1831): Hurst exponent via R/S analysis. H>0.5 trending, H<0.5 mean-reverting.
    - `__init__(self, period)` (line 1834): No method docstring.
    - `name(self)` (line 1838): No method docstring.
    - `calculate(self, df)` (line 1841): No method docstring.
  - `MeanReversionHalfLife` (line 1888): Ornstein-Uhlenbeck half-life via OLS. Lower = faster mean reversion.
    - `__init__(self, period)` (line 1891): No method docstring.
    - `name(self)` (line 1895): No method docstring.
    - `calculate(self, df)` (line 1898): No method docstring.
  - `ZScore` (line 1923): Z-Score: standardized deviation from rolling mean.
    - `__init__(self, period)` (line 1926): No method docstring.
    - `name(self)` (line 1930): No method docstring.
    - `calculate(self, df)` (line 1933): No method docstring.
  - `VarianceRatio` (line 1940): Lo-MacKinlay variance ratio. VR>1 = trending, VR<1 = mean-reverting.
    - `__init__(self, period, k)` (line 1943): No method docstring.
    - `name(self)` (line 1948): No method docstring.
    - `calculate(self, df)` (line 1951): No method docstring.
  - `Autocorrelation` (line 1959): Serial correlation of returns at lag k. Positive = momentum, negative = mean-reversion.
    - `__init__(self, period, lag)` (line 1962): No method docstring.
    - `name(self)` (line 1967): No method docstring.
    - `calculate(self, df)` (line 1970): No method docstring.
  - `KalmanTrend` (line 1988): 1D Kalman filter for price trend extraction.
    - `__init__(self, process_noise, measurement_noise)` (line 1991): No method docstring.
    - `name(self)` (line 1996): No method docstring.
    - `calculate(self, df)` (line 2001): No method docstring.
  - `ShannonEntropy` (line 2033): Shannon entropy of return distribution. High = uncertain, low = predictable.
    - `__init__(self, period, n_bins)` (line 2036): No method docstring.
    - `name(self)` (line 2041): No method docstring.
    - `calculate(self, df)` (line 2044): No method docstring.
  - `ApproximateEntropy` (line 2060): Approximate Entropy (ApEn). Low = regular/predictable, high = complex/random.
    - `__init__(self, period, m, r_mult)` (line 2063): No method docstring.
    - `name(self)` (line 2069): No method docstring.
    - `calculate(self, df)` (line 2072): No method docstring.
  - `AmihudIlliquidity` (line 2108): Amihud illiquidity ratio: |return| / dollar_volume. Higher = less liquid.
    - `__init__(self, period)` (line 2111): No method docstring.
    - `name(self)` (line 2115): No method docstring.
    - `calculate(self, df)` (line 2118): No method docstring.
  - `KyleLambda` (line 2125): Kyle's lambda price impact coefficient via rolling regression.
    - `__init__(self, period)` (line 2128): No method docstring.
    - `name(self)` (line 2132): No method docstring.
    - `calculate(self, df)` (line 2135): No method docstring.
  - `RollSpread` (line 2159): Roll's implied bid-ask spread in basis points.
    - `__init__(self, period)` (line 2162): No method docstring.
    - `name(self)` (line 2166): No method docstring.
    - `calculate(self, df)` (line 2169): No method docstring.
  - `FractalDimension` (line 2182): Higuchi fractal dimension. D~1 = smooth/trending, D~2 = rough/noisy.
    - `__init__(self, period, k_max)` (line 2185): No method docstring.
    - `name(self)` (line 2190): No method docstring.
    - `calculate(self, df)` (line 2193): No method docstring.
  - `DFA` (line 2235): Detrended Fluctuation Analysis. alpha>0.5 = persistent, alpha<0.5 = anti-persistent.
    - `__init__(self, period)` (line 2238): No method docstring.
    - `name(self)` (line 2242): No method docstring.
    - `calculate(self, df)` (line 2245): No method docstring.
  - `DominantCycle` (line 2308): FFT-based dominant cycle period in bars.
    - `__init__(self, period)` (line 2311): No method docstring.
    - `name(self)` (line 2315): No method docstring.
    - `calculate(self, df)` (line 2318): No method docstring.
  - `ReturnSkewness` (line 2359): Rolling skewness of returns. Negative = left tail risk.
    - `__init__(self, period)` (line 2362): No method docstring.
    - `name(self)` (line 2366): No method docstring.
    - `calculate(self, df)` (line 2369): No method docstring.
  - `ReturnKurtosis` (line 2373): Rolling excess kurtosis. High = fat tails (tail risk).
    - `__init__(self, period)` (line 2376): No method docstring.
    - `name(self)` (line 2380): No method docstring.
    - `calculate(self, df)` (line 2383): No method docstring.
  - `CUSUMDetector` (line 2391): CUSUM change-point detection. Output = bars since last regime change / period.
    - `__init__(self, period, threshold)` (line 2394): No method docstring.
    - `name(self)` (line 2399): No method docstring.
    - `calculate(self, df)` (line 2402): No method docstring.
  - `RegimePersistence` (line 2437): Consecutive bars in the same trend regime (price vs SMA).
    - `__init__(self, period)` (line 2440): No method docstring.
    - `name(self)` (line 2444): No method docstring.
    - `calculate(self, df)` (line 2447): No method docstring.
- Top-level functions:
  - `get_all_indicators()` (line 2475): Return dictionary of all indicator classes.
  - `create_indicator(name, kwargs)` (line 2630): Create an indicator by name with given parameters.

## Package `kalshi`

### `kalshi/__init__.py`
- Lines: 58
- Module intent: Kalshi vertical for intraday event-market research.
- Imports (12): `.client`, `.storage`, `.provider`, `.pipeline`, `.router`, `.quality`, `.mapping_store`, `.distribution`, `.options`, `.promotion`, `.events`, `.walkforward`
- Classes: none
- Top-level functions: none

### `kalshi/client.py`
- Lines: 630
- Module intent: Kalshi API client with signed authentication, rate limiting, and endpoint routing.
- Imports (14): `__future__`, `base64`, `logging`, `os`, `subprocess`, `tempfile`, `threading`, `time`, `dataclasses`, `datetime`, `typing`, `urllib.parse`, `requests`, `.router`
- Classes:
  - `RetryPolicy` (line 31): No class docstring.
    - Methods: none
  - `RateLimitPolicy` (line 38): No class docstring.
    - Methods: none
  - `RequestLimiter` (line 43): Lightweight token-bucket limiter with runtime limit updates.
    - `__init__(self, policy)` (line 48): No method docstring.
    - `_refill(self)` (line 56): No method docstring.
    - `acquire(self)` (line 62): No method docstring.
    - `update_rate(self, requests_per_second, burst)` (line 73): No method docstring.
    - `update_from_account_limits(self, payload)` (line 80): Attempt to derive limiter settings from API account-limit payloads.
  - `KalshiSigner` (line 119): Signs Kalshi requests using RSA-PSS SHA256.
    - `__init__(self, access_key, private_key_path, private_key_pem, passphrase, sign_func)` (line 131): No method docstring.
    - `available(self)` (line 154): No method docstring.
    - `_canonical_path(path)` (line 158): No method docstring.
    - `_load_private_key(self)` (line 163): Load and cache the private key using the cryptography library.
    - `_sign_with_cryptography(self, message)` (line 183): Sign using the in-process cryptography library (preferred).
    - `_sign_with_openssl(self, message, key_path)` (line 199): Fallback: sign using OpenSSL subprocess.
    - `sign(self, timestamp_ms, method, path)` (line 240): No method docstring.
  - `KalshiClient` (line 281): Kalshi HTTP wrapper with:
    - `__init__(self, base_url, environment, historical_base_url, historical_cutoff_ts, access_key, private_key_path, private_key_pem, private_key_passphrase, signer, retry_policy, rate_limit_policy, limiter, router, session, logger, api_key, api_secret)` (line 295): No method docstring.
    - `available(self)` (line 376): No method docstring.
    - `_join_url(base_url, path)` (line 380): No method docstring.
    - `_auth_headers(self, method, signed_path)` (line 393): No method docstring.
    - `_request_with_retries(self, method, url, signed_path, params, json_body)` (line 408): No method docstring.
    - `_request(self, method, path, params, json_body)` (line 464): No method docstring.
    - `get(self, path, params)` (line 497): No method docstring.
    - `paginate(self, path, params, items_key, cursor_key, cursor_param)` (line 500): Iterate API pages until no next cursor is returned.
    - `get_account_limits(self)` (line 526): No method docstring.
    - `fetch_historical_cutoff(self)` (line 531): No method docstring.
    - `server_time_utc(self)` (line 544): Fetch server time when endpoint exists; otherwise return local UTC now.
    - `clock_skew_seconds(self)` (line 564): No method docstring.
    - `list_markets(self, status)` (line 569): No method docstring.
    - `list_contracts(self, market_id)` (line 582): No method docstring.
    - `list_trades(self, contract_id, start_ts, end_ts)` (line 592): No method docstring.
    - `list_quotes(self, contract_id, start_ts, end_ts)` (line 612): No method docstring.
- Top-level functions:
  - `_normalize_env(value)` (line 23): No function docstring.

### `kalshi/disagreement.py`
- Lines: 112
- Module intent: Cross-market disagreement engine for Kalshi event features.
- Imports (5): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `DisagreementSignals` (line 22): No class docstring.
    - Methods: none
- Top-level functions:
  - `compute_disagreement(kalshi_entropy, kalshi_tail_mass, kalshi_variance, options_iv, options_skew, entropy_history, tail_history)` (line 31): Compute cross-market disagreement signals.
  - `disagreement_as_feature_dict(signals)` (line 103): Convert disagreement signals to a flat dict for feature panel merging.

### `kalshi/distribution.py`
- Lines: 917
- Module intent: Contract -> probability distribution builder for Kalshi markets.
- Imports (6): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`, `.quality`
- Classes:
  - `DistributionConfig` (line 28): No class docstring.
    - Methods: none
  - `DirectionResult` (line 118): Result of threshold direction resolution with confidence metadata.
    - Methods: none
  - `BinValidationResult` (line 182): Result of bin overlap/gap/ordering validation.
    - Methods: none
- Top-level functions:
  - `_is_tz_aware_datetime(series)` (line 22): No function docstring.
  - `_to_utc_timestamp(value)` (line 61): No function docstring.
  - `_prob_from_mid(mid, price_scale)` (line 68): No function docstring.
  - `_entropy(p)` (line 81): No function docstring.
  - `_isotonic_nonincreasing(y)` (line 89): Pool-adjacent-violators for nonincreasing constraints.
  - `_isotonic_nondecreasing(y)` (line 113): No function docstring.
  - `_resolve_threshold_direction(row)` (line 125): Resolve threshold contract semantics:
  - `_resolve_threshold_direction_with_confidence(row)` (line 135): Resolve threshold contract semantics with confidence scoring.
  - `_validate_bins(contracts)` (line 190): Validate bin structure for non-overlapping, ordered coverage.
  - `_tail_thresholds(event_type, config, support)` (line 251): No function docstring.
  - `_latest_quotes_asof(quotes, asof_ts, stale_minutes)` (line 266): No function docstring.
  - `_normalize_mass(mass)` (line 285): No function docstring.
  - `_moments(support, mass)` (line 294): No function docstring.
  - `_cdf_from_pmf(support, mass)` (line 310): No function docstring.
  - `_pmf_on_grid(support, mass, grid)` (line 320): No function docstring.
  - `_distribution_distances(support_a, mass_a, support_b, mass_b)` (line 327): No function docstring.
  - `_tail_probs_from_mass(support, mass, thresholds)` (line 366): No function docstring.
  - `_tail_probs_from_threshold_curve(thresholds_x, prob_curve, direction, fixed_thresholds)` (line 384): No function docstring.
  - `_estimate_liquidity_proxy(quotes, asof_ts)` (line 410): Estimate a stable liquidity proxy from recent quote stream.
  - `build_distribution_snapshot(contracts, quotes, asof_ts, config, event_ts, event_type)` (line 447): No function docstring.
  - `_lag_slug(lag)` (line 803): No function docstring.
  - `_add_distance_features(panel, lags)` (line 807): No function docstring.
  - `build_distribution_panel(markets, contracts, quotes, snapshot_times, config)` (line 850): Build market-level distribution snapshots across times.

### `kalshi/events.py`
- Lines: 511
- Module intent: Event-time joins and as-of feature/label builders for Kalshi-driven research.
- Imports (5): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `EventTimestampMeta` (line 14): Authoritative event timestamp metadata (D2).
    - Methods: none
  - `EventFeatureConfig` (line 22): No class docstring.
    - Methods: none
- Top-level functions:
  - `_to_utc_ts(series)` (line 29): No function docstring.
  - `_ensure_asof_before_release(df)` (line 33): No function docstring.
  - `asof_join(left, right, by, left_ts_col, right_ts_col)` (line 41): Strict backward as-of join (no forward-peeking).
  - `build_event_snapshot_grid(macro_events, horizons)` (line 76): No function docstring.
  - `_merge_event_market_map(grid, event_market_map)` (line 108): No function docstring.
  - `_add_revision_speed_features(joined, feature_cols)` (line 143): No function docstring.
  - `add_reference_disagreement_features(event_panel, reference_features, ts_col)` (line 180): Optional cross-market disagreement block via strict backward as-of join.
  - `build_event_feature_panel(macro_events, event_market_map, kalshi_distributions, config)` (line 222): Build event-centric panel indexed by (event_id, asof_ts).
  - `build_asset_time_feature_panel(asset_frame, kalshi_distributions, market_id, asof_col)` (line 319): Optional continuous panel keyed by (asset_id, ts) with strict as-of joins.
  - `build_event_labels(macro_events, event_outcomes_first_print, event_outcomes_revised, label_mode)` (line 353): Build event outcome labels with explicit as-of awareness.
  - `build_asset_response_labels(macro_events, asset_prices, ts_col, price_col, window_start, window_end, entry_delay, exit_horizon, price_source)` (line 438): Event-to-asset response labels with realistic execution windows.

### `kalshi/mapping_store.py`
- Lines: 70
- Module intent: Versioned event-to-market mapping persistence.
- Imports (5): `__future__`, `dataclasses`, `typing`, `pandas`, `.storage`
- Classes:
  - `EventMarketMappingRecord` (line 15): No class docstring.
    - Methods: none
  - `EventMarketMappingStore` (line 25): No class docstring.
    - `__init__(self, store)` (line 26): No method docstring.
    - `upsert(self, rows)` (line 29): No method docstring.
    - `asof(self, asof_ts)` (line 48): No method docstring.
    - `current_version(self)` (line 51): Return the latest mapping_version string from the store.
    - `assert_consistent_mapping_version(panel)` (line 61): Raise ValueError if a panel contains mixed mapping versions (D1).
- Top-level functions: none

### `kalshi/microstructure.py`
- Lines: 126
- Module intent: Market microstructure diagnostics for Kalshi event markets.
- Imports (5): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `MicrostructureDiagnostics` (line 20): No class docstring.
    - Methods: none
- Top-level functions:
  - `compute_microstructure(quotes, window_hours, asof_ts)` (line 30): Compute microstructure diagnostics from a quote panel.
  - `microstructure_as_feature_dict(diag)` (line 116): Convert diagnostics to a flat dict for feature panel merging.

### `kalshi/options.py`
- Lines: 144
- Module intent: OptionMetrics-style options reference features for Kalshi event disagreement.
- Imports (5): `__future__`, `typing`, `numpy`, `pandas`, `..features.options_factors`
- Classes: none
- Top-level functions:
  - `_to_utc_ts(series)` (line 17): No function docstring.
  - `build_options_reference_panel(options_frame, ts_col)` (line 21): Build a normalized options reference panel with:
  - `add_options_disagreement_features(event_panel, options_reference, options_ts_col)` (line 71): Strict backward as-of join of options reference features into event panel.

### `kalshi/pipeline.py`
- Lines: 158
- Module intent: Orchestration helpers for the Kalshi event-market vertical.
- Imports (14): `__future__`, `dataclasses`, `datetime`, `pathlib`, `typing`, `pandas`, `.distribution`, `.events`, `.options`, `.promotion`, `.provider`, `.storage`, `.walkforward`, `..autopilot.promotion_gate`
- Classes:
  - `KalshiPipeline` (line 29): No class docstring.
    - `from_store(cls, db_path, backend, provider)` (line 34): No method docstring.
    - `sync_reference(self, status, mapping_version)` (line 46): No method docstring.
    - `sync_intraday_quotes(self, start_ts, end_ts)` (line 63): No method docstring.
    - `build_distributions(self, start_ts, end_ts, freq, config)` (line 80): No method docstring.
    - `build_event_features(self, event_market_map, asof_ts, config, options_reference, options_ts_col)` (line 95): No method docstring.
    - `run_walkforward(self, event_features, labels, config, label_col)` (line 122): No method docstring.
    - `evaluate_walkforward_contract(self, walkforward_result, n_bootstrap, max_events_per_day)` (line 136): No method docstring.
    - `evaluate_event_promotion(self, walkforward_result, promotion_config, extra_contract_metrics)` (line 148): No method docstring.
- Top-level functions: none

### `kalshi/promotion.py`
- Lines: 176
- Module intent: Event-strategy promotion helpers for Kalshi walk-forward outputs.
- Imports (9): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`, `..autopilot.promotion_gate`, `..autopilot.strategy_discovery`, `..backtest.engine`, `.walkforward`
- Classes:
  - `EventPromotionConfig` (line 22): No class docstring.
    - Methods: none
- Top-level functions:
  - `_to_backtest_result(event_returns, horizon_days)` (line 32): No function docstring.
  - `evaluate_event_promotion(walkforward_result, config, gate, extra_contract_metrics)` (line 125): Evaluate Kalshi event strategy promotion from walk-forward outputs.

### `kalshi/provider.py`
- Lines: 628
- Module intent: Kalshi provider: ingestion + storage + feature-ready retrieval.
- Imports (12): `__future__`, `hashlib`, `json`, `datetime`, `typing`, `numpy`, `pandas`, `..config`, `.client`, `.distribution`, `.mapping_store`, `.storage`
- Classes:
  - `KalshiProvider` (line 67): Provider interface similar to WRDSProvider, but for event-market data.
    - `__init__(self, client, store)` (line 72): No method docstring.
    - `available(self)` (line 89): No method docstring.
    - `sync_account_limits(self)` (line 92): No method docstring.
    - `refresh_historical_cutoff(self)` (line 100): No method docstring.
    - `sync_market_catalog(self, status, mapping_version, mapping_source)` (line 108): No method docstring.
    - `sync_contracts(self, market_ids)` (line 219): No method docstring.
    - `sync_quotes(self, contract_ids, start_ts, end_ts)` (line 311): No method docstring.
    - `get_markets(self)` (line 406): No method docstring.
    - `get_contracts(self)` (line 411): No method docstring.
    - `get_quotes(self, market_id, start_ts, end_ts)` (line 416): No method docstring.
    - `get_event_market_map_asof(self, asof_ts)` (line 443): No method docstring.
    - `get_macro_events(self, versioned)` (line 448): No method docstring.
    - `get_event_outcomes(self, table)` (line 455): No method docstring.
    - `compute_and_store_distributions(self, start_ts, end_ts, freq, config)` (line 469): No method docstring.
    - `materialize_daily_health_report(self)` (line 550): Build and persist daily Kalshi ingestion/coverage health aggregates.
    - `get_daily_health_report(self)` (line 614): No method docstring.
    - `store_clock_check(self)` (line 621): No method docstring.
- Top-level functions:
  - `_to_iso_utc(value)` (line 40): No function docstring.
  - `_safe_hash_text(text)` (line 54): No function docstring.
  - `_asof_date(value)` (line 58): No function docstring.

### `kalshi/quality.py`
- Lines: 203
- Module intent: Quality scoring helpers for Kalshi event-distribution snapshots.
- Imports (4): `__future__`, `dataclasses`, `typing`, `numpy`
- Classes:
  - `QualityDimensions` (line 16): No class docstring.
    - Methods: none
  - `StalePolicy` (line 26): No class docstring.
    - Methods: none
- Top-level functions:
  - `_finite(values)` (line 48): No function docstring.
  - `dynamic_stale_cutoff_minutes(time_to_event_minutes, policy, market_type, liquidity_proxy)` (line 53): Dynamic stale-cutoff schedule:
  - `compute_quality_dimensions(expected_contracts, observed_contracts, spreads, quote_ages_seconds, volumes, open_interests, violation_magnitude)` (line 101): Multi-dimensional quality model for distribution snapshots.
  - `passes_hard_gates(quality, stale_cutoff_seconds, min_coverage, max_median_spread)` (line 161): Hard validity gates (C1).  Must-pass criteria — failing any gate means
  - `quality_as_feature_dict(quality)` (line 190): Expose soft quality dimensions as separate learnable feature columns (C2).

### `kalshi/regimes.py`
- Lines: 141
- Module intent: Regime tagging for Kalshi event strategies.
- Imports (5): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `EventRegimeTag` (line 21): No class docstring.
    - Methods: none
- Top-level functions:
  - `classify_inflation_regime(cpi_yoy, high_threshold, low_threshold)` (line 27): Classify inflation regime from CPI year-over-year.
  - `classify_policy_regime(fed_funds_change_bps, tightening_threshold_bps, easing_threshold_bps)` (line 42): Classify monetary policy regime from Fed funds rate changes.
  - `classify_vol_regime(vix_level, high_threshold, low_threshold)` (line 57): Classify volatility regime from VIX level.
  - `tag_event_regime(cpi_yoy, fed_funds_change_bps, vix_level)` (line 72): Tag an event with macro regime classifications.
  - `evaluate_strategy_by_regime(event_returns, regime_tags)` (line 85): Evaluate strategy performance breakdown by regime.
  - `regime_stability_score(breakdown)` (line 119): Score strategy stability across regimes (0-1).

### `kalshi/router.py`
- Lines: 95
- Module intent: Routing helpers for live vs historical Kalshi endpoints.
- Imports (5): `__future__`, `dataclasses`, `typing`, `urllib.parse`, `pandas`
- Classes:
  - `RouteDecision` (line 14): No class docstring.
    - Methods: none
  - `KalshiDataRouter` (line 20): Chooses live vs historical endpoint roots by cutoff timestamp.
    - `__init__(self, live_base_url, historical_base_url, historical_cutoff_ts, historical_prefix)` (line 25): No method docstring.
    - `_to_utc_ts(value)` (line 42): No method docstring.
    - `update_cutoff(self, cutoff_ts)` (line 50): No method docstring.
    - `_extract_end_ts(self, params)` (line 53): No method docstring.
    - `_clean_path(path)` (line 63): No method docstring.
    - `resolve(self, path, params)` (line 69): No method docstring.
- Top-level functions: none

### `kalshi/storage.py`
- Lines: 623
- Module intent: Event-time storage layer for Kalshi + macro event research.
- Imports (7): `__future__`, `json`, `sqlite3`, `datetime`, `pathlib`, `typing`, `pandas`
- Classes:
  - `EventTimeStore` (line 35): Intraday/event-time storage with a stable schema.
    - `__init__(self, db_path, backend)` (line 44): No method docstring.
    - `_execute(self, sql, params)` (line 66): No method docstring.
    - `_executemany(self, sql, rows)` (line 83): No method docstring.
    - `_table_columns(self, table)` (line 97): No method docstring.
    - `_norm_ts(value)` (line 111): No method docstring.
    - `_clean_value(key, value)` (line 125): No method docstring.
    - `_insert_or_replace(self, table, rows)` (line 137): No method docstring.
    - `init_schema(self)` (line 161): No method docstring.
    - `upsert_markets(self, rows)` (line 490): No method docstring.
    - `upsert_contracts(self, rows)` (line 493): No method docstring.
    - `append_quotes(self, rows)` (line 496): No method docstring.
    - `upsert_macro_events(self, rows)` (line 499): No method docstring.
    - `upsert_event_outcomes(self, rows)` (line 538): No method docstring.
    - `upsert_event_outcomes_first_print(self, rows)` (line 545): No method docstring.
    - `upsert_event_outcomes_revised(self, rows)` (line 552): No method docstring.
    - `upsert_distributions(self, rows)` (line 558): No method docstring.
    - `upsert_event_market_map_versions(self, rows)` (line 561): No method docstring.
    - `append_market_specs(self, rows)` (line 564): No method docstring.
    - `append_contract_specs(self, rows)` (line 567): No method docstring.
    - `upsert_data_provenance(self, rows)` (line 570): No method docstring.
    - `upsert_coverage_diagnostics(self, rows)` (line 573): No method docstring.
    - `upsert_ingestion_logs(self, rows)` (line 576): No method docstring.
    - `upsert_daily_health_reports(self, rows)` (line 579): No method docstring.
    - `upsert_ingestion_checkpoints(self, rows)` (line 582): No method docstring.
    - `get_ingestion_checkpoint(self, market_id, asof_date, endpoint)` (line 585): Return checkpoint row if this market/date/endpoint was already ingested.
    - `get_event_market_map_asof(self, asof_ts)` (line 598): No method docstring.
    - `query_df(self, sql, params)` (line 612): No method docstring.
- Top-level functions: none

### `kalshi/tests/__init__.py`
- Lines: 1
- Module intent: Kalshi package-local tests.
- Imports: none
- Classes: none
- Top-level functions: none

### `kalshi/tests/test_bin_validity.py`
- Lines: 105
- Module intent: Bin overlap/gap detection test (Instructions I.3).
- Imports (3): `unittest`, `pandas`, `quant_engine.kalshi.distribution`
- Classes:
  - `BinValidityTests` (line 13): Tests for bin overlap/gap/ordering validation.
    - `test_clean_bins_valid(self)` (line 16): Contiguous, non-overlapping, ordered bins should pass.
    - `test_overlapping_bins_detected(self)` (line 28): Overlapping bins should be flagged.
    - `test_gapped_bins_detected(self)` (line 38): Gaps between bins should be measured.
    - `test_inverted_bin_detected(self)` (line 49): bin_low >= bin_high should be flagged.
    - `test_single_bin_valid(self)` (line 59): A single bin should always be valid.
    - `test_missing_columns_valid(self)` (line 69): DataFrame without bin columns should return valid (no check needed).
    - `test_empty_dataframe_valid(self)` (line 75): Empty DataFrame should return valid.
    - `test_unordered_bins_detected(self)` (line 81): Bins in wrong order should have support_is_ordered=False after sorting.
    - `test_severe_overlap(self)` (line 93): Completely overlapping bins should be caught.
- Top-level functions: none

### `kalshi/tests/test_distribution.py`
- Lines: 36
- Module intent: No module docstring.
- Imports (3): `unittest`, `pandas`, `quant_engine.kalshi.distribution`
- Classes:
  - `DistributionLocalTests` (line 8): No class docstring.
    - `test_bin_distribution_probability_mass_is_normalized(self)` (line 9): No method docstring.
- Top-level functions: none

### `kalshi/tests/test_leakage.py`
- Lines: 41
- Module intent: No module docstring.
- Imports (3): `unittest`, `pandas`, `quant_engine.kalshi.events`
- Classes:
  - `LeakageLocalTests` (line 8): No class docstring.
    - `test_feature_rows_strictly_pre_release(self)` (line 9): No method docstring.
- Top-level functions: none

### `kalshi/tests/test_no_leakage.py`
- Lines: 117
- Module intent: No-leakage test at panel level (Instructions I.4).
- Imports (4): `unittest`, `numpy`, `pandas`, `quant_engine.kalshi.events`
- Classes:
  - `NoLeakageTests` (line 16): Panel-level look-ahead bias detection.
    - `_build_synthetic_panel(self, n_events)` (line 19): Build a synthetic panel to check for temporal leakage.
    - `test_all_asof_before_release(self)` (line 60): Every panel row must have asof_ts strictly before release_ts.
    - `test_single_event_no_leakage(self)` (line 80): Even a single event must have asof_ts < release_ts.
- Top-level functions: none

### `kalshi/tests/test_signature_kat.py`
- Lines: 141
- Module intent: Known-answer test for Kalshi RSA-PSS SHA256 signature (Instructions A3 + I.1).
- Imports (3): `base64`, `unittest`, `quant_engine.kalshi.client`
- Classes:
  - `SignatureKATTests` (line 46): Known-answer tests for Kalshi request signing.
    - `_skip_if_no_crypto(self)` (line 49): No method docstring.
    - `test_sign_produces_valid_base64(self)` (line 55): Signing should produce a valid base64-encoded signature string.
    - `test_sign_deterministic_message_format(self)` (line 72): The canonical message format must be <ts><METHOD><path>.
    - `test_sign_verifies_with_public_key(self)` (line 89): Signature must verify against the corresponding public key.
    - `test_canonical_path_normalization(self)` (line 131): Path should be normalized to canonical form.
- Top-level functions: none

### `kalshi/tests/test_stale_quotes.py`
- Lines: 152
- Module intent: Stale quote cutoff test (Instructions I.5).
- Imports (2): `unittest`, `quant_engine.kalshi.quality`
- Classes:
  - `StaleQuoteCutoffTests` (line 15): Tests for dynamic stale-cutoff schedule.
    - `test_near_event_tight_cutoff(self)` (line 18): Within near_event_minutes of event, cutoff should be near_event_stale_minutes.
    - `test_far_event_loose_cutoff(self)` (line 31): Beyond far_event_minutes, cutoff should be far_event_stale_minutes.
    - `test_midpoint_interpolation(self)` (line 44): Between near and far, cutoff should interpolate.
    - `test_cutoff_monotonically_increases_with_distance(self)` (line 60): Cutoff should increase as we get farther from the event.
    - `test_cpi_market_type_multiplier(self)` (line 75): CPI market type should tighten the cutoff (multiplier < 1).
    - `test_fomc_market_type_multiplier(self)` (line 88): FOMC should have the tightest multiplier.
    - `test_low_liquidity_widens_cutoff(self)` (line 101): Low liquidity should widen the cutoff (multiplier > 1).
    - `test_high_liquidity_tightens_cutoff(self)` (line 115): High liquidity should tighten the cutoff (multiplier < 1).
    - `test_none_time_to_event_uses_base(self)` (line 129): None time_to_event should use base_stale_minutes.
    - `test_cutoff_clamped_to_bounds(self)` (line 137): Cutoff should be clamped between min and max.
- Top-level functions: none

### `kalshi/tests/test_threshold_direction.py`
- Lines: 126
- Module intent: Threshold direction correctness test (Instructions I.2).
- Imports (2): `unittest`, `quant_engine.kalshi.distribution`
- Classes:
  - `ThresholdDirectionTests` (line 15): Tests for threshold direction resolution.
    - `test_explicit_ge_direction(self)` (line 20): No method docstring.
    - `test_explicit_le_direction(self)` (line 26): No method docstring.
    - `test_explicit_gte_alias(self)` (line 32): No method docstring.
    - `test_explicit_lte_alias(self)` (line 37): No method docstring.
    - `test_explicit_ge_symbol(self)` (line 42): No method docstring.
    - `test_explicit_le_symbol(self)` (line 46): No method docstring.
    - `test_payout_structure_above(self)` (line 50): No method docstring.
    - `test_payout_structure_below(self)` (line 56): No method docstring.
    - `test_rules_text_greater_than(self)` (line 64): No method docstring.
    - `test_rules_text_less_than(self)` (line 71): No method docstring.
    - `test_rules_text_above(self)` (line 78): No method docstring.
    - `test_rules_text_below(self)` (line 84): No method docstring.
    - `test_title_guess_or_higher(self)` (line 92): No method docstring.
    - `test_title_guess_or_lower(self)` (line 99): No method docstring.
    - `test_no_direction_signal(self)` (line 108): No method docstring.
    - `test_empty_row(self)` (line 113): No method docstring.
    - `test_legacy_resolve_returns_string(self)` (line 119): No method docstring.
- Top-level functions: none

### `kalshi/tests/test_walkforward_purge.py`
- Lines: 159
- Module intent: Walk-forward purge/embargo test (Instructions I.6).
- Imports (4): `unittest`, `numpy`, `pandas`, `quant_engine.kalshi.walkforward`
- Classes:
  - `WalkForwardPurgeTests` (line 18): Tests that walk-forward purge/embargo prevents data leakage.
    - `_build_synthetic_data(self, n_events)` (line 21): Build synthetic panel + labels for walk-forward testing.
    - `test_no_train_events_in_purge_window(self)` (line 43): Training events must not fall within purge window of test events.
    - `test_event_type_aware_purge(self)` (line 94): Event-type-aware purge should use the larger window for mixed events.
    - `test_embargo_removes_adjacent_events(self)` (line 111): Embargo should remove events adjacent to the train/test boundary.
    - `test_trial_counting(self)` (line 142): n_trials_total should be positive and reflect the search space.
- Top-level functions: none

### `kalshi/walkforward.py`
- Lines: 477
- Module intent: Walk-forward evaluation for event-centric Kalshi feature panels.
- Imports (6): `__future__`, `dataclasses`, `typing`, `numpy`, `pandas`, `..backtest.advanced_validation`
- Classes:
  - `EventWalkForwardConfig` (line 16): No class docstring.
    - Methods: none
  - `EventWalkForwardFold` (line 35): No class docstring.
    - Methods: none
  - `EventWalkForwardResult` (line 50): No class docstring.
    - `wf_oos_corr(self)` (line 59): No method docstring.
    - `wf_positive_fold_fraction(self)` (line 66): No method docstring.
    - `wf_is_oos_gap(self)` (line 72): No method docstring.
    - `worst_event_loss(self)` (line 79): No method docstring.
    - `to_metrics(self)` (line 84): No method docstring.
- Top-level functions:
  - `_bootstrap_mean_ci(values, n_bootstrap, random_seed)` (line 95): No function docstring.
  - `_event_regime_stability(event_returns, event_types)` (line 109): No function docstring.
  - `evaluate_event_contract_metrics(result, n_bootstrap, max_events_per_day)` (line 137): Advanced validation contract metrics for event strategies:
  - `_corr(a, b)` (line 228): No function docstring.
  - `_fit_ridge(X, y, alpha)` (line 241): No function docstring.
  - `_predict(X, beta)` (line 250): No function docstring.
  - `_prepare_panel(panel, labels, label_col)` (line 254): No function docstring.
  - `run_event_walkforward(panel, labels, config, label_col)` (line 301): No function docstring.

## Package `models`

### `models/__init__.py`
- Lines: 20
- Module intent: Models subpackage — training, prediction, versioning, and retraining triggers.
- Imports (6): `.governance`, `.cross_sectional`, `.calibration`, `.neural_net`, `.walk_forward`, `.feature_stability`
- Classes: none
- Top-level functions: none

### `models/calibration.py`
- Lines: 216
- Module intent: Confidence Calibration --- Platt scaling and isotonic regression.
- Imports (3): `__future__`, `typing`, `numpy`
- Classes:
  - `_LinearRescaler` (line 41): Maps raw scores to [0, 1] via min-max linear rescaling.
    - `__init__(self)` (line 47): No method docstring.
    - `fit(self, raw, _outcomes)` (line 51): No method docstring.
    - `transform(self, raw)` (line 57): No method docstring.
  - `ConfidenceCalibrator` (line 66): Post-hoc confidence calibration via Platt scaling or isotonic regression.
    - `__init__(self, method)` (line 81): No method docstring.
    - `fit(self, raw_confidence, actual_outcomes)` (line 92): Fit the calibration mapping.
    - `_fit_sklearn(self, raw, outcomes)` (line 129): Fit using sklearn implementations.
    - `transform(self, raw_confidence)` (line 153): Transform raw confidence scores into calibrated probabilities.
    - `fit_transform(self, raw_confidence, actual_outcomes)` (line 189): Fit on the data and return calibrated scores in one step.
    - `is_fitted(self)` (line 199): Whether the calibrator has been fitted.
    - `backend(self)` (line 204): Return which backend is in use: 'sklearn' or 'linear_fallback'.
    - `__repr__(self)` (line 210): No method docstring.
- Top-level functions: none

### `models/cross_sectional.py`
- Lines: 136
- Module intent: Cross-Sectional Ranking Model — rank stocks relative to peers at each date.
- Imports (3): `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `cross_sectional_rank(predictions, date_col, prediction_col, asset_col, long_quantile, short_quantile)` (line 18): Rank stocks cross-sectionally by predicted return at each date.

### `models/feature_stability.py`
- Lines: 311
- Module intent: Feature Stability Monitoring — tracks feature importance rankings across
- Imports (7): `json`, `time`, `dataclasses`, `pathlib`, `typing`, `numpy`, `..config`
- Classes:
  - `StabilityReport` (line 47): Summary returned by :meth:`FeatureStabilityTracker.check_stability`.
    - `to_dict(self)` (line 60): No method docstring.
  - `FeatureStabilityTracker` (line 68): Record and compare feature importance rankings over training cycles.
    - `__init__(self, history_path, top_n, shift_threshold)` (line 82): No method docstring.
    - `_load(self)` (line 100): Load existing history from disk, if present.
    - `_save(self)` (line 115): Persist current history to disk.
    - `record_importance(self, cycle_id, feature_importances, feature_names)` (line 129): Store the top-N feature importance ranking for a training cycle.
    - `_spearman_rank_correlation(ranking_a, importances_a, ranking_b, importances_b)` (line 174): Compute Spearman rank correlation between two importance rankings.
    - `check_stability(self)` (line 215): Analyse the recorded history and return a stability report.
- Top-level functions: none

### `models/governance.py`
- Lines: 108
- Module intent: Champion/challenger governance for model versions.
- Imports (6): `json`, `dataclasses`, `datetime`, `pathlib`, `typing`, `..config`
- Classes:
  - `ChampionRecord` (line 14): No class docstring.
    - `to_dict(self)` (line 21): No method docstring.
  - `ModelGovernance` (line 25): Maintains champion model per horizon and promotes challengers if better.
    - `__init__(self, registry_path)` (line 30): No method docstring.
    - `_load(self)` (line 34): No method docstring.
    - `_save(self, payload)` (line 40): No method docstring.
    - `_score(metrics)` (line 45): No method docstring.
    - `get_champion_version(self, horizon)` (line 51): No method docstring.
    - `evaluate_and_update(self, horizon, version_id, metrics, min_relative_improvement)` (line 58): No method docstring.
- Top-level functions: none

### `models/iv/__init__.py`
- Lines: 31
- Module intent: Implied Volatility Surface Models — Heston, SVI, Black-Scholes, and IV Surface.
- Imports (1): `.models`
- Classes: none
- Top-level functions: none

### `models/iv/models.py`
- Lines: 928
- Module intent: Implied Volatility Surface Models.
- Imports (5): `dataclasses`, `enum`, `typing`, `numpy`, `scipy`
- Classes:
  - `OptionType` (line 19): No class docstring.
    - Methods: none
  - `Greeks` (line 25): Option Greeks container.
    - Methods: none
  - `HestonParams` (line 35): Heston model parameters.
    - `validate(self)` (line 43): Check Feller condition: 2*kappa*theta > sigma^2.
  - `SVIParams` (line 49): Raw SVI parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).
    - Methods: none
  - `BlackScholes` (line 58): Black-Scholes option pricing and analytics.
    - `price(S, K, T, r, sigma, q, option_type)` (line 62): European option price via Black-Scholes.
    - `greeks(S, K, T, r, sigma, q, option_type)` (line 80): Compute all Greeks.
    - `implied_vol(price, S, K, T, r, q, option_type, tol, max_iter)` (line 110): Solve for implied volatility using Brent's method with Newton warm-start.
    - `iv_surface(S, strikes, expiries, r, q, base_vol)` (line 131): Generate a BS IV surface (flat, used as baseline reference).
  - `HestonModel` (line 138): Heston (1993) stochastic volatility model.
    - `__init__(self, params)` (line 150): No method docstring.
    - `characteristic_function(self, u, T, r, q)` (line 153): Heston characteristic function phi(u) for log-spot.
    - `price(self, S, K, T, r, q, option_type)` (line 172): Price European option via numerical integration of characteristic function.
    - `implied_vol(self, S, K, T, r, q, option_type)` (line 212): Compute BS-equivalent implied vol from Heston price.
    - `iv_surface(self, S, strikes, expiries, r, q)` (line 217): Generate the Heston implied volatility surface.
    - `calibrate(self, market_ivs, strikes, expiries, S, r, q)` (line 246): Calibrate Heston parameters to market implied volatilities.
  - `SVIModel` (line 293): SVI (Stochastic Volatility Inspired) implied variance parameterization.
    - `__init__(self, params)` (line 304): No method docstring.
    - `total_variance(self, k)` (line 307): Compute total implied variance w(k) for log-moneyness k.
    - `implied_vol(self, k, T)` (line 312): Compute implied volatility from SVI total variance.
    - `iv_surface(self, S, strikes, expiries, r, q)` (line 318): Generate the SVI implied volatility surface.
    - `smile(self, T, S, r, q, n_strikes)` (line 339): Generate a single smile curve for expiry T.
    - `calibrate(self, market_ivs, strikes, T, S, r, q)` (line 347): Calibrate SVI parameters to a single expiry smile.
    - `check_no_butterfly_arbitrage(self)` (line 380): Verify no-butterfly-arbitrage conditions on SVI parameters.
  - `ArbitrageFreeSVIBuilder` (line 394): Arbitrage-aware SVI surface builder.
    - `__init__(self, penalty_weight, calendar_epsilon, max_iter)` (line 405): No method docstring.
    - `_svi_total_variance(k, params)` (line 416): No method docstring.
    - `_initial_guess(k, target_w)` (line 422): No method docstring.
    - `_vega_spread_weights(strikes, ivs, expiry, spot, r, q)` (line 436): Approximate bid-ask/vega weighting when quotes are unavailable.
    - `_slice_objective(self, x, k, target_w, weights)` (line 459): No method docstring.
    - `fit_slice(self, strikes, ivs, expiry, spot, r, q)` (line 487): Fit one SVI smile using weighted total-variance loss.
    - `enforce_calendar_monotonicity(total_variance, eps)` (line 544): Force non-decreasing total variance in maturity for each strike.
    - `interpolate_total_variance(expiries, total_variance, query_expiries)` (line 554): Linear maturity interpolation in total variance space.
    - `build_surface(self, spot, strikes, expiries, market_iv_grid, r, q)` (line 573): Calibrate an arbitrage-aware SVI surface from market IV quotes.
  - `IVPoint` (line 681): Single implied-volatility observation.
    - Methods: none
  - `IVSurface` (line 691): Store and interpolate an implied-volatility surface.
    - `__init__(self, spot, r, q)` (line 708): No method docstring.
    - `add_point(self, strike, expiry, iv, kwargs)` (line 720): Add a single IV observation.
    - `add_slice(self, strikes, expiry, ivs)` (line 725): Add an entire smile (one expiry, many strikes).
    - `add_surface(self, strikes, expiries, iv_grid)` (line 732): Add a full grid (n_expiry x n_strike) of IVs at once.
    - `n_points(self)` (line 747): No method docstring.
    - `_log_moneyness(self, K, T)` (line 754): Convert strike to log forward-moneyness.
    - `_build_interpolator(self)` (line 759): Rebuild the 2-D interpolator from stored points.
    - `get_iv(self, strike, expiry)` (line 783): Interpolate IV at an arbitrary (strike, expiry) point.
    - `get_smile(self, expiry, strikes, n_points)` (line 797): Return an interpolated smile for a given expiry.
    - `decompose(self, expiry, strikes, n_points)` (line 824): Extract Karhunen-Loeve-inspired principal modes from a smile.
    - `decompose_surface(self, expiries)` (line 902): Run ``decompose`` across multiple expiries.
- Top-level functions:
  - `generate_synthetic_market_surface(S, r, q)` (line 648): Generate a realistic synthetic market IV surface for demonstration.

### `models/neural_net.py`
- Lines: 197
- Module intent: Tabular Neural Network — feedforward network for tabular financial data.
- Imports (2): `typing`, `numpy`
- Classes:
  - `TabularNet` (line 29): Feedforward network for tabular financial data.
    - `__init__(self, input_dim, hidden_dims, dropout, lr, batch_size, epochs)` (line 36): No method docstring.
    - `_build_model(self)` (line 59): Construct the PyTorch Sequential model.
    - `fit(self, X, y, sample_weight, verbose)` (line 76): Train the network on tabular features.
    - `predict(self, X)` (line 147): Generate predictions for the given features.
    - `feature_importances_(self)` (line 182): Approximate feature importance from first-layer weights.
- Top-level functions: none

### `models/predictor.py`
- Lines: 375
- Module intent: Model Predictor — loads trained ensemble and generates predictions.
- Imports (9): `json`, `pathlib`, `typing`, `joblib`, `numpy`, `pandas`, `..config`, `.governance`, `.versioning`
- Classes:
  - `EnsemblePredictor` (line 45): Loads a trained regime-conditional ensemble and generates predictions.
    - `__init__(self, horizon, model_dir, version)` (line 60): No method docstring.
    - `_resolve_model_dir(self)` (line 71): Resolve the actual directory containing model files.
    - `_load(self)` (line 114): Load all model artifacts from disk.
    - `predict(self, features, regimes, regime_confidence, regime_probabilities)` (line 159): Generate predictions for all rows.
    - `predict_single(self, features, regime, regime_confidence, regime_probabilities)` (line 332): Predict for a single observation (e.g., latest bar).
- Top-level functions:
  - `_prepare_features(raw, expected, medians)` (line 23): Align, impute, and return features matching expected column order.

### `models/retrain_trigger.py`
- Lines: 296
- Module intent: ML Retraining Trigger Logic
- Imports (6): `json`, `os`, `datetime`, `typing`, `numpy`, `..config`
- Classes:
  - `RetrainTrigger` (line 37): Determines when ML model should be retrained.
    - `__init__(self, metadata_file, max_days_between_retrain, min_new_trades_for_retrain, min_acceptable_win_rate, min_acceptable_spearman, min_acceptable_sharpe, min_acceptable_ic, lookback_trades)` (line 40): Args:
    - `_load_metadata(self)` (line 73): Load model metadata.
    - `_save_metadata(self)` (line 87): Save model metadata.
    - `add_trade_result(self, is_winner, net_return, predicted_return, actual_return)` (line 93): Record a completed trade result for performance monitoring.
    - `check(self)` (line 136): Check all retraining triggers.
    - `record_retraining(self, n_trades, oos_spearman, version_id, notes)` (line 226): Record that retraining has been completed.
    - `status(self)` (line 258): Human-readable status summary.
- Top-level functions: none

### `models/trainer.py`
- Lines: 1340
- Module intent: Model Trainer — trains regime-conditional gradient boosting ensemble.
- Imports (11): `json`, `time`, `dataclasses`, `pathlib`, `typing`, `joblib`, `numpy`, `pandas`, `..config`, `.feature_stability`, `.versioning`
- Classes:
  - `IdentityScaler` (line 85): No-op scaler that passes data through unchanged.
    - `fit(self, X, y)` (line 92): No method docstring.
    - `transform(self, X)` (line 95): No method docstring.
    - `fit_transform(self, X, y)` (line 98): No method docstring.
    - `inverse_transform(self, X)` (line 102): No method docstring.
  - `DiverseEnsemble` (line 106): Lightweight ensemble wrapper that combines predictions from multiple
    - `__init__(self, estimators, weights)` (line 114): Args:
    - `_aggregate_feature_importances(self)` (line 137): Weighted average of feature_importances_ across estimators that expose it.
    - `predict(self, X)` (line 155): Return the weighted average of all constituent model predictions.
  - `TrainResult` (line 165): Result of training a single model.
    - Methods: none
  - `EnsembleResult` (line 184): Result of training the full regime-conditional ensemble.
    - Methods: none
  - `ModelTrainer` (line 194): Trains a regime-conditional gradient boosting ensemble for
    - `__init__(self, model_params, max_features, cv_folds, holdout_fraction, max_gap)` (line 200): No method docstring.
    - `_spearmanr(x, y)` (line 215): Spearman correlation with optional SciPy dependency.
    - `_require_sklearn()` (line 239): Fail fast when training is requested without scikit-learn.
    - `_extract_dates(index)` (line 248): Return row-aligned timestamps from an index (supports panel MultiIndex).
    - `_sort_panel_by_time(cls, X, y, sample_weights)` (line 257): Sort panel rows by date (then security id) to enforce deterministic chronology.
    - `_temporal_holdout_masks(dates, holdout_fraction, min_dev_rows, min_hold_rows)` (line 278): Split by unique dates (not raw rows) to avoid cross-asset temporal leakage.
    - `_date_purged_folds(dates, n_folds, purge_gap, embargo)` (line 312): Generate expanding-window folds using unique dates for panel-safe CV.
    - `_prune_correlated_features(X, threshold)` (line 350): Remove highly correlated features before permutation importance.
    - `_select_features(self, X_train, y_train, max_feats)` (line 380): Feature selection: correlation pruning then permutation importance.
    - `train_ensemble(self, features, targets, regimes, regime_probabilities, horizon, verbose, versioned, survivorship_mode, universe_as_of, recency_weight)` (line 446): Train the full regime-conditional ensemble.
    - `_train_single(self, X, y, model_name, horizon, max_features_override, verbose, sample_weights)` (line 624): Train a single gradient boosting model with feature selection.
    - `_train_diverse_ensemble(self, X_dev_sel, y_dev, w_dev, gbr_model, gbr_scaler, verbose)` (line 845): Train ElasticNet, RandomForest, and optionally XGBoost and LightGBM
    - `_optimize_ensemble_weights(self, estimators, X_dev_sel, y_dev, w_dev, model_names, verbose)` (line 976): Find optimal stacking weights by minimising MSE on OOS predictions
    - `_clone_model(model)` (line 1135): Create a fresh (unfitted) clone of a model with the same hyperparameters.
    - `_make_model(self, use_early_stopping)` (line 1156): Create a fresh GradientBoostingRegressor with configured params.
    - `_save(self, global_model, global_scaler, global_features, global_result, regime_models, regime_scalers, regime_feature_sets, regime_results, horizon, ensemble_result, versioned, survivorship_mode, universe_as_of, train_data_start, train_data_end)` (line 1173): Save all model artifacts to disk using joblib (safe serialization).
    - `_print_summary(self, result, regimes, targets)` (line 1273): Print comprehensive training summary with gate diagnostics.
- Top-level functions: none

### `models/versioning.py`
- Lines: 204
- Module intent: Model Versioning — timestamped model directories with registry.
- Imports (7): `json`, `shutil`, `dataclasses`, `datetime`, `pathlib`, `typing`, `..config`
- Classes:
  - `ModelVersion` (line 26): Metadata for a single model version.
    - `to_dict(self)` (line 47): No method docstring.
    - `from_dict(cls, d)` (line 51): No method docstring.
  - `ModelRegistry` (line 58): Manages versioned model storage and retrieval.
    - `__init__(self, model_dir)` (line 65): No method docstring.
    - `_load_registry(self)` (line 70): Load or initialize the registry.
    - `_save_registry(self)` (line 77): Persist registry to disk.
    - `latest_version_id(self)` (line 84): Get the latest version ID, or None if no versions exist.
    - `get_latest(self)` (line 88): Get the latest model version metadata.
    - `get_version(self, version_id)` (line 95): Get metadata for a specific version.
    - `get_version_dir(self, version_id)` (line 102): Get the directory path for a model version.
    - `get_latest_dir(self)` (line 106): Get the directory path for the latest version.
    - `list_versions(self)` (line 113): List all versions, newest first.
    - `create_version_dir(self, version_id)` (line 118): Create a new timestamped version directory.
    - `register_version(self, version)` (line 132): Register a newly trained model version and set it as latest.
    - `rollback(self, version_id)` (line 153): Set a previous version as the active latest.
    - `prune_old(self, keep_n)` (line 168): Remove old versions beyond keep_n most recent.
    - `has_versions(self)` (line 202): Check if any versions are registered.
- Top-level functions: none

### `models/walk_forward.py`
- Lines: 235
- Module intent: Walk-Forward Model Selection — expanding-window hyperparameter search
- Imports (4): `itertools`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `_spearmanr(x, y)` (line 24): Spearman rank correlation (scipy optional).
  - `_expanding_walk_forward_folds(dates, n_folds, horizon)` (line 45): Generate expanding-window walk-forward folds using unique dates.
  - `_extract_dates(index)` (line 84): Return row-aligned timestamps from an index (supports panel MultiIndex).
  - `walk_forward_select(features, targets, regimes, param_grid, n_folds, horizon)` (line 93): Select the best model configuration via walk-forward cross-validation.

## Package `regime`

### `regime/__init__.py`
- Lines: 14
- Module intent: Regime modeling components.
- Imports (3): `.correlation`, `.detector`, `.hmm`
- Classes: none
- Top-level functions: none

### `regime/correlation.py`
- Lines: 212
- Module intent: Correlation Regime Detection (NEW 11).
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes:
  - `CorrelationRegimeDetector` (line 29): Detect regime changes in pairwise correlation structure.
    - `__init__(self, window, z_score_lookback, threshold)` (line 45): No method docstring.
    - `compute_rolling_correlation(self, returns_dict, window)` (line 60): Compute rolling average pairwise correlation across the universe.
    - `detect_correlation_spike(self, threshold)` (line 122): Flag dates where average correlation exceeds the threshold.
    - `get_correlation_features(self, returns_dict, window, threshold, z_score_lookback)` (line 153): Return a DataFrame with all correlation regime features.
- Top-level functions: none

### `regime/detector.py`
- Lines: 292
- Module intent: Regime detector with two engines:
- Imports (7): `dataclasses`, `typing`, `logging`, `numpy`, `pandas`, `..config`, `.hmm`
- Classes:
  - `RegimeOutput` (line 35): No class docstring.
    - Methods: none
  - `RegimeDetector` (line 43): Classifies market regime at each bar using either rules or HMM.
    - `__init__(self, method, hurst_trend_threshold, hurst_mr_threshold, adx_trend_threshold, vol_spike_quantile, vol_lookback, hmm_states, hmm_max_iter, hmm_stickiness, min_duration)` (line 48): No method docstring.
    - `detect(self, features)` (line 73): No method docstring.
    - `_rule_detect(self, features)` (line 76): No method docstring.
    - `_hmm_detect(self, features)` (line 123): No method docstring.
    - `detect_with_confidence(self, features)` (line 186): No method docstring.
    - `detect_full(self, features)` (line 190): No method docstring.
    - `regime_features(self, features)` (line 195): Generate regime-derived features for the ML model.
    - `_get_col(df, col, default)` (line 246): No method docstring.
- Top-level functions:
  - `detect_regimes_batch(features_by_id, detector, verbose)` (line 252): Shared regime detection across multiple PERMNOs.

### `regime/hmm.py`
- Lines: 501
- Module intent: Gaussian HMM regime model with sticky transitions and duration smoothing.
- Imports (4): `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `HMMFitResult` (line 23): No class docstring.
    - Methods: none
  - `GaussianHMM` (line 30): Gaussian HMM using EM (Baum-Welch).
    - `__init__(self, n_states, max_iter, min_covar, stickiness, random_state, min_duration, prior_weight, covariance_type)` (line 39): No method docstring.
    - `_ensure_positive_definite(self, cov)` (line 66): Verify positive-definiteness via Cholesky; if it fails, add
    - `_init_params(self, X)` (line 83): No method docstring.
    - `_log_emission(self, X)` (line 137): No method docstring.
    - `_forward_backward(self, log_emit)` (line 169): No method docstring.
    - `viterbi(self, X)` (line 209): Return the most likely state sequence via the Viterbi algorithm.
    - `_smooth_duration(self, states, probs)` (line 251): Merge very short runs into neighboring states (HSMM-like smoothing).
    - `fit(self, X)` (line 285): No method docstring.
    - `predict_proba(self, X)` (line 343): No method docstring.
- Top-level functions:
  - `_logsumexp(a, axis)` (line 13): No function docstring.
  - `select_hmm_states_bic(X, min_states, max_states, hmm_kwargs)` (line 351): Select the optimal number of HMM states using the Bayesian Information Criterion.
  - `build_hmm_observation_matrix(features)` (line 430): Build a robust, low-dimensional observation matrix for regime inference.
  - `map_raw_states_to_regimes(raw_states, features)` (line 452): Map unlabeled HMM states -> semantic regimes used by the system.

## Package `risk`

### `risk/__init__.py`
- Lines: 42
- Module intent: Risk Management Module — Renaissance-grade portfolio risk controls.
- Imports (10): `.position_sizer`, `.portfolio_risk`, `.drawdown`, `.metrics`, `.stop_loss`, `.covariance`, `.factor_portfolio`, `.portfolio_optimizer`, `.attribution`, `.stress_test`
- Classes: none
- Top-level functions: none

### `risk/attribution.py`
- Lines: 266
- Module intent: Performance Attribution --- decompose portfolio returns into market, factor, and alpha.
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `_estimate_beta(portfolio_returns, benchmark_returns)` (line 26): OLS beta of portfolio vs benchmark.
  - `_estimate_factor_loadings(portfolio_returns, benchmark_returns, factor_returns)` (line 49): Multivariate OLS regression of excess returns on factor returns.
  - `decompose_returns(portfolio_returns, benchmark_returns, factor_returns)` (line 87): Decompose portfolio returns into market, factor, and alpha components.
  - `compute_attribution_report(portfolio_returns, benchmark_returns, factor_returns, annual_trading_days)` (line 178): Produce an extended attribution summary with risk-adjusted metrics.

### `risk/covariance.py`
- Lines: 244
- Module intent: Covariance estimation utilities for portfolio risk controls.
- Imports (4): `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `CovarianceEstimate` (line 12): No class docstring.
    - Methods: none
  - `CovarianceEstimator` (line 18): Estimate a robust covariance matrix for asset returns.
    - `__init__(self, method, shrinkage, annualization)` (line 23): No method docstring.
    - `estimate(self, returns)` (line 33): No method docstring.
    - `portfolio_volatility(self, weights, covariance)` (line 57): No method docstring.
    - `_estimate_values(self, values)` (line 75): No method docstring.
- Top-level functions:
  - `compute_regime_covariance(returns, regimes, min_obs, shrinkage)` (line 119): Compute separate covariance matrices for each market regime.
  - `get_regime_covariance(regime_covs, current_regime)` (line 206): Return the covariance matrix for *current_regime*.

### `risk/drawdown.py`
- Lines: 233
- Module intent: Drawdown Controller — circuit breakers and recovery protocols.
- Imports (5): `dataclasses`, `enum`, `typing`, `numpy`, `pandas`
- Classes:
  - `DrawdownState` (line 19): No class docstring.
    - Methods: none
  - `DrawdownStatus` (line 28): Current drawdown state and action directives.
    - Methods: none
  - `DrawdownController` (line 43): Multi-tier drawdown protection with circuit breakers.
    - `__init__(self, warning_threshold, caution_threshold, critical_threshold, daily_loss_limit, weekly_loss_limit, recovery_days, initial_equity)` (line 58): No method docstring.
    - `update(self, daily_pnl)` (line 85): Update equity and return current drawdown status.
    - `_compute_actions(self, state, drawdown)` (line 173): Determine size multiplier, entry permission, and liquidation flag.
    - `reset(self, equity)` (line 193): Reset controller state.
    - `get_summary(self)` (line 204): Get drawdown controller summary statistics.
- Top-level functions: none

### `risk/factor_portfolio.py`
- Lines: 220
- Module intent: Factor-Based Portfolio Construction — factor decomposition and exposure analysis.
- Imports (3): `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `compute_factor_exposures(returns, factor_returns)` (line 18): Estimate factor betas for each asset via OLS regression.
  - `compute_residual_returns(returns, factor_returns, factor_betas)` (line 135): Strip out systematic factor exposure, returning idiosyncratic returns.

### `risk/metrics.py`
- Lines: 251
- Module intent: Risk Metrics — VaR, CVaR, tail risk, MAE/MFE, and advanced risk analytics.
- Imports (4): `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `RiskReport` (line 15): Comprehensive risk metrics report.
    - Methods: none
  - `RiskMetrics` (line 49): Computes comprehensive risk metrics from trade returns and equity curves.
    - `__init__(self, annual_trading_days)` (line 54): No method docstring.
    - `compute_full_report(self, trade_returns, equity_curve, trade_details, holding_days)` (line 57): Compute all risk metrics.
    - `_drawdown_analytics(self, equity)` (line 144): Compute drawdown metrics from a daily equity curve.
    - `_drawdown_analytics_array(self, equity)` (line 149): Compute drawdown metrics from an equity array.
    - `_compute_mae_mfe(self, trade_details)` (line 187): Compute Maximum Adverse/Favorable Excursion from intra-trade data.
    - `_empty_report(self)` (line 216): No method docstring.
    - `print_report(self, report)` (line 225): Pretty-print a risk report.
- Top-level functions: none

### `risk/portfolio_optimizer.py`
- Lines: 255
- Module intent: Mean-Variance Portfolio Optimization — turnover-penalised portfolio construction.
- Imports (4): `typing`, `numpy`, `pandas`, `scipy.optimize`
- Classes: none
- Top-level functions:
  - `optimize_portfolio(expected_returns, covariance, current_weights, max_position, max_portfolio_vol, turnover_penalty, risk_aversion, sector_map, max_sector_exposure)` (line 21): Find optimal portfolio weights via mean-variance optimization.

### `risk/portfolio_risk.py`
- Lines: 327
- Module intent: Portfolio Risk Manager — enforces sector, correlation, and exposure limits.
- Imports (6): `dataclasses`, `typing`, `numpy`, `pandas`, `..config`, `.covariance`
- Classes:
  - `RiskCheck` (line 51): Result of a portfolio risk check.
    - Methods: none
  - `PortfolioRiskManager` (line 58): Enforces portfolio-level risk constraints.
    - `__init__(self, max_sector_pct, max_corr_between, max_gross_exposure, max_single_name_pct, max_beta_exposure, max_portfolio_vol, correlation_lookback, covariance_method, sector_map)` (line 66): No method docstring.
    - `_infer_ticker_from_price_df(df)` (line 90): No method docstring.
    - `_resolve_sector(self, asset_id, price_data)` (line 102): Resolve sector for a PERMNO-first key, falling back to ticker metadata.
    - `check_new_position(self, ticker, position_size, current_positions, price_data, benchmark_data)` (line 115): Check if adding a new position violates any risk constraints.
    - `_check_correlations(self, new_ticker, existing_tickers, price_data)` (line 206): Find max correlation between new ticker and existing positions.
    - `_estimate_portfolio_beta(self, positions, price_data, benchmark_data)` (line 233): Estimate portfolio beta vs benchmark.
    - `_estimate_portfolio_vol(self, positions, price_data)` (line 262): Estimate annualized portfolio volatility from covariance matrix.
    - `portfolio_summary(self, positions, price_data)` (line 291): Generate a portfolio risk summary.
- Top-level functions: none

### `risk/position_sizer.py`
- Lines: 290
- Module intent: Position Sizing — Kelly criterion, volatility-scaled, and ATR-based methods.
- Imports (4): `dataclasses`, `typing`, `numpy`, `pandas`
- Classes:
  - `PositionSize` (line 18): Result of position sizing calculation.
    - Methods: none
  - `PositionSizer` (line 31): Multi-method position sizer with conservative blending.
    - `__init__(self, target_portfolio_vol, max_position_pct, min_position_pct, max_risk_per_trade, kelly_fraction, atr_multiplier, blend_weights)` (line 49): No method docstring.
    - `size_position(self, ticker, win_rate, avg_win, avg_loss, realized_vol, atr, price, holding_days, confidence, n_current_positions, max_positions)` (line 71): Calculate position size using multiple methods and blend.
    - `_kelly(self, win_rate, avg_win, avg_loss)` (line 151): Kelly criterion: f* = (p*b - q) / b
    - `_vol_scaled(self, realized_vol, holding_days, max_positions)` (line 171): Volatility-targeted sizing.
    - `_atr_based(self, atr, price)` (line 192): ATR-based sizing: risk a fixed fraction of capital.
    - `size_portfolio(self, signals, price_data, trade_history, max_positions)` (line 211): Size all candidate positions for the portfolio.
- Top-level functions: none

### `risk/stop_loss.py`
- Lines: 251
- Module intent: Stop Loss Manager — regime-aware ATR stops, trailing, time, and regime-change stops.
- Imports (6): `dataclasses`, `enum`, `typing`, `numpy`, `pandas`, `..config`
- Classes:
  - `StopReason` (line 26): No class docstring.
    - Methods: none
  - `StopResult` (line 37): Result of stop-loss evaluation.
    - Methods: none
  - `StopLossManager` (line 48): Multi-strategy stop-loss manager.
    - `__init__(self, hard_stop_pct, atr_stop_multiplier, trailing_atr_multiplier, trailing_activation_pct, max_holding_days, regime_change_exit, profit_target_pct)` (line 65): No method docstring.
    - `evaluate(self, entry_price, current_price, highest_price, atr, bars_held, entry_regime, current_regime)` (line 83): Evaluate all stop conditions for an open position.
    - `compute_initial_stop(self, entry_price, atr, regime)` (line 219): Compute the initial stop-loss price for a new position.
    - `compute_risk_per_share(self, entry_price, atr, regime)` (line 237): Compute risk per share based on regime-adjusted stop distance.
- Top-level functions: none

### `risk/stress_test.py`
- Lines: 363
- Module intent: Stress Testing Module --- scenario analysis and historical drawdown replay.
- Imports (4): `__future__`, `typing`, `numpy`, `pandas`
- Classes: none
- Top-level functions:
  - `_estimate_portfolio_beta(portfolio_weights, returns_history, min_obs)` (line 66): Estimate weighted-average beta of the portfolio vs equal-weight market proxy.
  - `_compute_portfolio_vol(portfolio_weights, returns_history, annual_trading_days)` (line 111): Annualized portfolio volatility from historical covariance.
  - `run_stress_scenarios(portfolio_weights, returns_history, scenarios)` (line 136): Apply stress scenarios to a portfolio and estimate impact.
  - `run_historical_drawdown_test(portfolio_weights, returns_history, n_worst, min_drawdown_pct)` (line 207): Replay the worst historical drawdown episodes on the portfolio.
  - `_find_drawdown_episodes(returns, min_drawdown_pct)` (line 303): Identify non-overlapping drawdown episodes from a return series.

## Package `tests`

### `tests/__init__.py`
- Lines: 2
- Module intent: No module docstring.
- Imports: none
- Classes: none
- Top-level functions: none

### `tests/test_autopilot_predictor_fallback.py`
- Lines: 53
- Module intent: No module docstring.
- Imports (4): `unittest`, `unittest.mock`, `pandas`, `quant_engine.autopilot.engine`
- Classes:
  - `AutopilotPredictorFallbackTests` (line 9): No class docstring.
    - `test_ensure_predictor_falls_back_when_model_import_fails(self)` (line 10): No method docstring.
- Top-level functions: none

### `tests/test_cache_metadata_rehydrate.py`
- Lines: 91
- Module intent: No module docstring.
- Imports (6): `json`, `tempfile`, `unittest`, `pathlib`, `pandas`, `quant_engine.data.local_cache`
- Classes:
  - `CacheMetadataRehydrateTests` (line 26): No class docstring.
    - `test_rehydrate_writes_metadata_for_daily_csv(self)` (line 27): No method docstring.
    - `test_rehydrate_only_missing_does_not_overwrite(self)` (line 49): No method docstring.
    - `test_rehydrate_force_with_overwrite_source_updates_source(self)` (line 68): No method docstring.
- Top-level functions:
  - `_write_daily_csv(path)` (line 11): No function docstring.

### `tests/test_covariance_estimator.py`
- Lines: 20
- Module intent: No module docstring.
- Imports (4): `unittest`, `numpy`, `pandas`, `quant_engine.risk.covariance`
- Classes:
  - `CovarianceEstimatorTests` (line 9): No class docstring.
    - `test_single_asset_covariance_is_2d_and_positive(self)` (line 10): No method docstring.
- Top-level functions: none

### `tests/test_delisting_total_return.py`
- Lines: 71
- Module intent: No module docstring.
- Imports (4): `unittest`, `numpy`, `pandas`, `quant_engine.features.pipeline`
- Classes:
  - `DelistingTotalReturnTests` (line 9): No class docstring.
    - `test_target_uses_total_return_when_available(self)` (line 10): No method docstring.
    - `test_indicator_values_unaffected_by_delist_return_columns(self)` (line 37): No method docstring.
- Top-level functions: none

### `tests/test_drawdown_liquidation.py`
- Lines: 128
- Module intent: No module docstring.
- Imports (6): `unittest`, `types`, `pandas`, `quant_engine.backtest.engine`, `quant_engine.risk.drawdown`, `quant_engine.risk.stop_loss`
- Classes:
  - `_FakePositionSizer` (line 11): No class docstring.
    - `size_position(self, kwargs)` (line 12): No method docstring.
  - `_FakeDrawdownController` (line 16): No class docstring.
    - `__init__(self)` (line 17): No method docstring.
    - `update(self, daily_pnl)` (line 20): No method docstring.
    - `get_summary(self)` (line 50): No method docstring.
  - `_FakeStopLossManager` (line 54): No class docstring.
    - `evaluate(self, kwargs)` (line 55): No method docstring.
  - `_FakePortfolioRisk` (line 67): No class docstring.
    - `check_new_position(self, kwargs)` (line 68): No method docstring.
  - `_FakeRiskMetrics` (line 72): No class docstring.
    - `compute_full_report(self, args, kwargs)` (line 73): No method docstring.
  - `DrawdownLiquidationTests` (line 83): No class docstring.
    - `test_critical_drawdown_forces_liquidation(self)` (line 84): No method docstring.
- Top-level functions: none

### `tests/test_execution_dynamic_costs.py`
- Lines: 43
- Module intent: No module docstring.
- Imports (2): `unittest`, `quant_engine.backtest.execution`
- Classes:
  - `ExecutionDynamicCostTests` (line 6): No class docstring.
    - `test_dynamic_costs_increase_under_stress(self)` (line 7): No method docstring.
- Top-level functions: none

### `tests/test_integration.py`
- Lines: 556
- Module intent: End-to-end integration tests for the quant engine pipeline.
- Imports (5): `sys`, `pathlib`, `numpy`, `pandas`, `pytest`
- Classes:
  - `TestFullPipelineSynthetic` (line 70): End-to-end test: data -> features -> regimes -> training -> prediction -> backtest.
    - `synthetic_data(self)` (line 74): No method docstring.
    - `pipeline_outputs(self, synthetic_data)` (line 78): Run the pipeline once and cache results for all tests in this class.
    - `test_features_shape(self, pipeline_outputs, synthetic_data)` (line 126): Features DataFrame has expected shape.
    - `test_targets_shape(self, pipeline_outputs)` (line 137): Targets DataFrame is aligned with features.
    - `test_regimes_aligned(self, pipeline_outputs)` (line 146): Regimes series is aligned with features index.
    - `test_pit_no_future_in_features(self, pipeline_outputs, synthetic_data)` (line 155): PIT semantics: features at date t use only data up to date t.
    - `test_pit_no_future_in_targets(self, pipeline_outputs, synthetic_data)` (line 172): PIT semantics: targets use forward returns (shift -h), so the last h
    - `test_training_produces_result(self, pipeline_outputs)` (line 185): ModelTrainer produces a valid EnsembleResult with small config.
  - `TestCvGapHardBlock` (line 229): Verify that the CV gap hard block rejects overfit models.
    - `test_cv_gap_hard_block(self)` (line 232): A trainer with max_gap=0 should reject any model with nonzero CV gap.
  - `TestRegime2Suppression` (line 304): Verify regime 2 gating suppresses trades.
    - `test_regime_2_suppression(self)` (line 307): Backtester should suppress entries when regime==2 and confidence > 0.5.
    - `test_regime_0_not_suppressed(self)` (line 360): Regime 0 (trending bull) signals should NOT be suppressed.
  - `TestCrossSectionalRanking` (line 412): Verify cross-sectional ranker produces valid output.
    - `test_cross_sectional_rank_basic(self)` (line 415): Basic ranking: stocks get percentile ranks within each date.
    - `test_cross_sectional_rank_multiindex(self)` (line 453): Ranking works with MultiIndex (permno, date) DataFrames.
    - `test_cross_sectional_rank_zscore_centered(self)` (line 491): Z-scores should be approximately mean-zero within each date.
    - `test_cross_sectional_rank_signals_count(self)` (line 523): Long/short signals respect quantile thresholds.
- Top-level functions:
  - `_generate_synthetic_ohlcv(n_stocks, n_days, seed)` (line 23): Generate synthetic OHLCV data for *n_stocks* over *n_days*.

### `tests/test_iv_arbitrage_builder.py`
- Lines: 33
- Module intent: No module docstring.
- Imports (3): `unittest`, `numpy`, `quant_engine.models.iv.models`
- Classes:
  - `ArbitrageFreeSVIBuilderTests` (line 8): No class docstring.
    - `test_build_surface_has_valid_shape_and_monotone_total_variance(self)` (line 9): No method docstring.
- Top-level functions: none

### `tests/test_kalshi_asof_features.py`
- Lines: 60
- Module intent: No module docstring.
- Imports (3): `unittest`, `pandas`, `quant_engine.kalshi.events`
- Classes:
  - `KalshiAsofFeatureTests` (line 8): No class docstring.
    - `test_event_feature_panel_uses_backward_asof_join(self)` (line 9): No method docstring.
    - `test_event_feature_panel_raises_when_required_columns_missing(self)` (line 43): No method docstring.
- Top-level functions: none

### `tests/test_kalshi_distribution.py`
- Lines: 109
- Module intent: No module docstring.
- Imports (3): `unittest`, `pandas`, `quant_engine.kalshi.distribution`
- Classes:
  - `KalshiDistributionTests` (line 12): No class docstring.
    - `test_bin_distribution_normalizes_and_computes_moments(self)` (line 13): No method docstring.
    - `test_threshold_distribution_applies_monotone_constraint(self)` (line 46): No method docstring.
    - `test_distribution_panel_accepts_tz_aware_snapshot_times(self)` (line 78): No method docstring.
- Top-level functions: none

### `tests/test_kalshi_hardening.py`
- Lines: 600
- Module intent: No module docstring.
- Imports (16): `base64`, `tempfile`, `unittest`, `pathlib`, `numpy`, `pandas`, `quant_engine.kalshi.client`, `quant_engine.kalshi.distribution`, `quant_engine.kalshi.events`, `quant_engine.kalshi.mapping_store`, `quant_engine.kalshi.options`, `quant_engine.kalshi.promotion`, `quant_engine.kalshi.provider`, `quant_engine.kalshi.quality`, `quant_engine.kalshi.storage`, `quant_engine.kalshi.walkforward`
- Classes:
  - `KalshiHardeningTests` (line 33): No class docstring.
    - `test_bin_distribution_mass_normalizes_to_one(self)` (line 34): No method docstring.
    - `test_threshold_direction_semantics_change_tail_probabilities(self)` (line 61): No method docstring.
    - `test_unknown_threshold_direction_marked_quality_low(self)` (line 107): No method docstring.
    - `test_dynamic_stale_cutoff_tightens_near_event(self)` (line 133): No method docstring.
    - `test_dynamic_stale_cutoff_adjusts_for_market_type_and_liquidity(self)` (line 177): No method docstring.
    - `test_quality_score_behaves_sensibly_on_synthetic_cases(self)` (line 204): No method docstring.
    - `test_event_panel_supports_event_id_mapping(self)` (line 225): No method docstring.
    - `test_event_labels_first_vs_latest(self)` (line 259): No method docstring.
    - `test_walkforward_runs_and_counts_trials(self)` (line 288): No method docstring.
    - `test_walkforward_contract_metrics_are_computed(self)` (line 320): No method docstring.
    - `test_event_promotion_flow_uses_walkforward_contract_metrics(self)` (line 357): No method docstring.
    - `test_options_disagreement_features_are_joined_asof(self)` (line 396): No method docstring.
    - `test_mapping_store_asof(self)` (line 428): No method docstring.
    - `test_store_ingestion_and_health_tables(self)` (line 460): No method docstring.
    - `test_provider_materializes_daily_health_report(self)` (line 502): No method docstring.
    - `test_signer_canonical_payload_and_header_fields(self)` (line 580): No method docstring.
- Top-level functions: none

### `tests/test_loader_and_predictor.py`
- Lines: 220
- Module intent: No module docstring.
- Imports (9): `tempfile`, `unittest`, `unittest.mock`, `pandas`, `quant_engine.data.loader`, `quant_engine.data.local_cache`, `quant_engine.data.local_cache`, `quant_engine.data.local_cache`, `quant_engine.models.predictor`
- Classes:
  - `_FakeWRDSProvider` (line 18): No class docstring.
    - `available(self)` (line 19): No method docstring.
    - `get_crsp_prices(self, tickers, start_date, end_date)` (line 22): No method docstring.
    - `get_crsp_prices_with_delistings(self, tickers, start_date, end_date)` (line 43): No method docstring.
    - `resolve_permno(self, ticker, as_of_date)` (line 67): No method docstring.
  - `_UnavailableWRDSProvider` (line 71): No class docstring.
    - `available(self)` (line 72): No method docstring.
  - `LoaderAndPredictorTests` (line 76): No class docstring.
    - `test_load_ohlcv_uses_wrds_contract_and_stable_columns(self)` (line 77): No method docstring.
    - `test_load_with_delistings_applies_delisting_return(self)` (line 86): No method docstring.
    - `test_predictor_explicit_version_does_not_silently_fallback(self)` (line 97): No method docstring.
    - `test_cache_load_reads_daily_csv_when_parquet_unavailable(self)` (line 102): No method docstring.
    - `test_cache_save_falls_back_to_csv_without_parquet_engine(self)` (line 121): No method docstring.
    - `test_trusted_wrds_cache_short_circuits_live_wrds(self)` (line 143): No method docstring.
    - `test_untrusted_cache_refreshes_from_wrds_and_sets_wrds_source(self)` (line 164): No method docstring.
    - `test_survivorship_fallback_prefers_cached_subset_when_wrds_unavailable(self)` (line 188): No method docstring.
- Top-level functions: none

### `tests/test_panel_split.py`
- Lines: 50
- Module intent: No module docstring.
- Imports (3): `unittest`, `pandas`, `quant_engine.models.trainer`
- Classes:
  - `PanelSplitTests` (line 8): No class docstring.
    - `test_holdout_mask_uses_dates_not_raw_rows(self)` (line 9): No method docstring.
    - `test_date_purged_folds_do_not_overlap(self)` (line 28): No method docstring.
- Top-level functions: none

### `tests/test_paper_trader_kelly.py`
- Lines: 120
- Module intent: No module docstring.
- Imports (8): `json`, `tempfile`, `unittest`, `pathlib`, `numpy`, `pandas`, `quant_engine.autopilot.paper_trader`, `quant_engine.autopilot.registry`
- Classes:
  - `PaperTraderKellyTests` (line 108): No class docstring.
    - `test_kelly_sizing_changes_position_size_with_bounds(self)` (line 109): No method docstring.
- Top-level functions:
  - `_mock_price_data()` (line 13): No function docstring.
  - `_seed_state(path)` (line 30): No function docstring.
  - `_run_cycle(use_kelly)` (line 58): No function docstring.

### `tests/test_promotion_contract.py`
- Lines: 95
- Module intent: No module docstring.
- Imports (5): `unittest`, `pandas`, `quant_engine.autopilot.promotion_gate`, `quant_engine.autopilot.strategy_discovery`, `quant_engine.backtest.engine`
- Classes:
  - `PromotionContractTests` (line 50): No class docstring.
    - `test_contract_fails_when_advanced_requirements_fail(self)` (line 51): No method docstring.
    - `test_contract_passes_when_all_checks_pass(self)` (line 73): No method docstring.
- Top-level functions:
  - `_candidate()` (line 10): No function docstring.
  - `_result()` (line 22): No function docstring.

### `tests/test_provider_registry.py`
- Lines: 24
- Module intent: No module docstring.
- Imports (3): `unittest`, `quant_engine.data.provider_registry`, `quant_engine.kalshi.client`
- Classes:
  - `ProviderRegistryTests` (line 7): No class docstring.
    - `test_registry_lists_core_providers(self)` (line 8): No method docstring.
    - `test_registry_rejects_unknown_provider(self)` (line 13): No method docstring.
    - `test_registry_can_construct_kalshi_provider(self)` (line 17): No method docstring.
- Top-level functions: none

### `tests/test_research_factors.py`
- Lines: 127
- Module intent: No module docstring.
- Imports (5): `unittest`, `numpy`, `pandas`, `quant_engine.features.pipeline`, `quant_engine.features.research_factors`
- Classes:
  - `ResearchFactorTests` (line 35): No class docstring.
    - `test_single_asset_research_features_exist(self)` (line 36): No method docstring.
    - `test_cross_asset_network_features_shape_and_bounds(self)` (line 54): No method docstring.
    - `test_cross_asset_factors_are_causally_lagged(self)` (line 92): No method docstring.
    - `test_pipeline_universe_includes_research_features(self)` (line 110): No method docstring.
- Top-level functions:
  - `_make_ohlcv(seed, periods, drift)` (line 14): No function docstring.

### `tests/test_survivorship_pit.py`
- Lines: 69
- Module intent: No module docstring.
- Imports (5): `tempfile`, `unittest`, `pathlib`, `pandas`, `quant_engine.data.survivorship`
- Classes:
  - `SurvivorshipPointInTimeTests` (line 13): No class docstring.
    - `test_filter_panel_by_point_in_time_universe(self)` (line 14): No method docstring.
- Top-level functions: none

### `tests/test_validation_and_risk_extensions.py`
- Lines: 101
- Module intent: No module docstring.
- Imports (5): `unittest`, `numpy`, `pandas`, `quant_engine.backtest.validation`, `quant_engine.risk.portfolio_risk`
- Classes:
  - `ValidationAndRiskExtensionTests` (line 27): No class docstring.
    - `test_cpcv_detects_positive_signal_quality(self)` (line 28): No method docstring.
    - `test_spa_passes_for_consistently_positive_signal_returns(self)` (line 50): No method docstring.
    - `test_portfolio_risk_rejects_high_projected_volatility(self)` (line 67): No method docstring.
- Top-level functions:
  - `_make_ohlcv(close)` (line 14): No function docstring.

## Package `utils`

### `utils/__init__.py`
- Lines: 1
- Module intent: No module docstring.
- Imports: none
- Classes: none
- Top-level functions: none

### `utils/logging.py`
- Lines: 436
- Module intent: Structured logging for the quant engine.
- Imports (9): `__future__`, `json`, `logging`, `sys`, `urllib.request`, `urllib.error`, `datetime`, `pathlib`, `typing`
- Classes:
  - `StructuredFormatter` (line 22): JSON formatter for machine-parseable log output.
    - `format(self, record)` (line 31): No method docstring.
  - `AlertHistory` (line 76): Persistent alert history with optional webhook notifications.
    - `__init__(self, history_file, webhook_url, max_history)` (line 105): No method docstring.
    - `_load(self)` (line 124): Load alert history from disk.
    - `_save(self, records)` (line 136): Save alert history to disk, pruning to max_history.
    - `record(self, message, severity, context)` (line 145): Record an alert event and optionally send a webhook notification.
    - `record_batch(self, alerts, severity, context)` (line 183): Record multiple alert events at once.
    - `query(self, last_n, alert_type, since)` (line 233): Query alert history.
    - `_notify_webhook(self, event)` (line 277): Send an alert event to the configured webhook URL.
  - `MetricsEmitter` (line 320): Emit key metrics on every cycle and check alert thresholds.
    - `__init__(self, logger_name, alert_history)` (line 337): No method docstring.
    - `emit_cycle_metrics(self, model_age_days, ic, rolling_sharpe, regime_distribution, turnover, execution_costs)` (line 345): Log a structured metrics payload for the current cycle.
    - `check_alerts(self, ic, sharpe, drawdown, regime_2_duration)` (line 384): Check alert thresholds, log, persist to history, and notify.
- Top-level functions:
  - `get_logger(name, level)` (line 45): Get a structured logger for the quant engine.
