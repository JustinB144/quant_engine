# Dash UI Reference

## Purpose

This is the active UI reference for `dash_ui` (Dash). It documents:
- registered pages/routes,
- page IDs,
- callback endpoints and I/O wiring,
- shared UI component modules.

Legacy `ui/` is removed and should not be used.

## UI Shell Components

- `dash_ui/app.py`: app composition, page integration, shell wiring
- `dash_ui/server.py`: launcher/runtime server wrapper
- `dash_ui/theme.py`: theme constants and figure styling helpers
- `dash_ui/assets/style.css`: global CSS styling
- `dash_ui/components/*`: reusable UI primitives and chart utilities

## Shared Component Modules

- `dash_ui/components/alert_banner.py`: Alert banner component for displaying system messages and warnings. Functions: `alert_banner`. Classes: none.
- `dash_ui/components/chart_utils.py`: Plotly chart factory functions for the Quant Engine Dashboard. Functions: `line_chart`, `area_chart`, `bar_chart`, `heatmap_chart`, `surface_3d`, `equity_curve`, `regime_timeline`, `dual_axis_chart`, `candlestick_chart`, `scatter_chart`, `radar_chart`, `histogram_chart`. Classes: none.
- `dash_ui/components/health_check_list.py`: Reusable health check display component. Functions: `health_check_item`, `health_check_list`. Classes: none.
- `dash_ui/components/metric_card.py`: Reusable KPI metric card component. Functions: `metric_card`. Classes: none.
- `dash_ui/components/regime_badge.py`: Regime state badge component for displaying market regime indicators. Functions: `regime_badge`. Classes: none.
- `dash_ui/components/sidebar.py`: Sidebar navigation component with active-state highlighting. Functions: `_nav_item`, `create_sidebar`, `update_active_nav`. Classes: none.
- `dash_ui/components/status_bar.py`: Bottom status bar component. Functions: `create_status_bar`. Classes: none.
- `dash_ui/components/trade_table.py`: Styled DataTable component for displaying trades with conditional formatting. Functions: `trade_table`. Classes: none.
- `dash_ui/components/page_header.py`: Reusable page header component with title, subtitle, and action buttons. Functions: `create_page_header`. Classes: none.

## Registered Pages Summary

- Pages detected: 10
- Callbacks detected: 34

## Pages and Callback Wiring

### `Dashboard`

- Route: `/`
- Order: `0`
- File: `dash_ui/pages/dashboard.py`
- Purpose: Dashboard -- Portfolio Intelligence Overview.
- Declared UI IDs (13): `card-30d-return`, `card-cv-gap`, `card-data-quality`, `card-portfolio-value`, `card-regime`, `card-retrain`, `card-sharpe`, `card-system-health`, `dashboard-data`, `dashboard-interval`, `dashboard-refresh-btn`, `dashboard-tab-content`, `dashboard-tabs`
- Callback functions (3):
  - `load_dashboard_data` (line 175): outputs [dashboard-data.data]; inputs [dashboard-interval.n_intervals, dashboard-refresh-btn.n_clicks]
  - `update_metric_cards` (line 400): outputs [card-portfolio-value.children, card-30d-return.children, card-sharpe.children, card-regime.children, card-retrain.children, card-cv-gap.children, card-data-quality.children, card-system-health.children, dashboard-last-updated.children]; inputs [dashboard-data.data]
  - `render_tab_content` (line 456): outputs [dashboard-tab-content.children]; inputs [dashboard-tabs.value, dashboard-data.data]

### `System Health`

- Route: `/system-health`
- Order: `1`
- File: `dash_ui/pages/system_health.py`
- Purpose: System Health Console -- comprehensive health assessment for the Quant Engine.
- Declared UI IDs (11): `hc-card-complexity`, `hc-card-data`, `hc-card-execution`, `hc-card-overall`, `hc-card-promotion`, `hc-card-wf`, `health-data`, `health-interval`, `health-refresh-btn`, `health-tab-content`, `health-tabs`
- Callback functions (3):
  - `load_health_data` (line 230): outputs [health-data.data]; inputs [health-interval.n_intervals, health-refresh-btn.n_clicks]
  - `update_health_cards` (line 285): outputs [hc-card-overall.children, hc-card-data.children, hc-card-promotion.children, hc-card-wf.children, hc-card-execution.children, hc-card-complexity.children]; inputs [health-data.data]
  - `render_health_tab` (line 344): outputs [health-tab-content.children]; inputs [health-tabs.value, health-data.data]

### `Data Explorer`

- Route: `/data-explorer`
- Order: `2`
- File: `dash_ui/pages/data_explorer.py`
- Purpose: Data Explorer -- OHLCV visualization and quality analysis.
- Declared UI IDs (15): `de-load-btn`, `de-loaded-data`, `de-price-chart`, `de-quality-body`, `de-quality-btn`, `de-quality-close-btn`, `de-quality-modal`, `de-selected-ticker`, `de-stats-bar`, `de-status-text`, `de-ticker-input`, `de-ticker-list-container`, `de-timeframe-dropdown`, `de-universe-dropdown`, `de-volume-chart`
- Callback functions (7):
  - `load_data` (line 363): outputs [de-loaded-data.data, de-status-text.children]; inputs [de-load-btn.n_clicks]; state [de-universe-dropdown.value, de-timeframe-dropdown.value, de-ticker-input.value, de-loaded-data.data]
  - `render_ticker_list` (line 418): outputs [de-ticker-list-container.children]; inputs [de-loaded-data.data, de-selected-ticker.data]
  - `select_ticker` (line 471): outputs [de-selected-ticker.data]; inputs [n_clicks]; state [de-loaded-data.data]
  - `update_price_chart` (line 489): outputs [de-price-chart.figure]; inputs [de-selected-ticker.data]; state [de-loaded-data.data]
  - `update_volume_chart` (line 571): outputs [de-volume-chart.figure]; inputs [de-selected-ticker.data]; state [de-loaded-data.data]
  - `update_stats_bar` (line 613): outputs [de-stats-bar.children]; inputs [de-selected-ticker.data]; state [de-loaded-data.data]
  - `toggle_quality_modal` (line 666): outputs [de-quality-modal.is_open, de-quality-body.children]; inputs [de-quality-btn.n_clicks, de-quality-close-btn.n_clicks]; state [de-quality-modal.is_open, de-loaded-data.data]

### `Model Lab`

- Route: `/model-lab`
- Order: `3`
- File: `dash_ui/pages/model_lab.py`
- Purpose: Model Lab -- Feature engineering, regime detection, and model training.
- Declared UI IDs (22): `ml-correlation-heatmap`, `ml-cv-results-chart`, `ml-feature-data`, `ml-feature-importance-chart`, `ml-features-status`, `ml-load-features-btn`, `ml-model-summary`, `ml-regime-data`, `ml-regime-probs`, `ml-regime-status`, `ml-regime-timeline`, `ml-run-regime-btn`, `ml-tabs`, `ml-train-btn`, `ml-train-feature-mode`, `ml-train-horizon`, `ml-train-options`, `ml-train-progress`, `ml-train-status`, `ml-train-universe`, `ml-training-result`, `ml-transition-matrix`
- Callback functions (3):
  - `load_features` (line 426): outputs [ml-feature-importance-chart.figure, ml-correlation-heatmap.figure, ml-features-status.children]; inputs [ml-load-features-btn.n_clicks]
  - `run_regime_detection` (line 576): outputs [ml-regime-timeline.figure, ml-regime-probs.figure, ml-transition-matrix.figure, ml-regime-status.children]; inputs [ml-run-regime-btn.n_clicks]
  - `train_model` (line 741): outputs [ml-cv-results-chart.figure, ml-model-summary.children, ml-train-progress.value, ml-train-status.children]; inputs [ml-train-btn.n_clicks]; state [ml-train-universe.value, ml-train-horizon.value, ml-train-feature-mode.value, ml-train-options.value]

### `Signal Desk`

- Route: `/signal-desk`
- Order: `4`
- File: `dash_ui/pages/signal_desk.py`
- Purpose: Signal Desk -- Prediction generation and signal ranking.
- Declared UI IDs (9): `sd-distribution-chart`, `sd-generate-btn`, `sd-horizon-dropdown`, `sd-scatter-chart`, `sd-signal-table`, `sd-signal-table-container`, `sd-signals-data`, `sd-status-text`, `sd-topn-input`
- Callback functions (1):
  - `generate_signals` (line 306): outputs [sd-signal-table-container.children, sd-distribution-chart.figure, sd-scatter-chart.figure, sd-status-text.children, sd-signals-data.data]; inputs [sd-generate-btn.n_clicks]; state [sd-horizon-dropdown.value, sd-topn-input.value]

### `Backtest & Risk`

- Route: `/backtest-risk`
- Order: `5`
- File: `dash_ui/pages/backtest_risk.py`
- Purpose: Backtest & Risk -- Equity curves, risk metrics, and trade analysis.
- Declared UI IDs (18): `bt-drawdown-chart`, `bt-entry-threshold`, `bt-equity-curve`, `bt-holding-period`, `bt-max-positions`, `bt-position-size`, `bt-regime-perf`, `bt-returns-store`, `bt-risk-mgmt`, `bt-run-btn`, `bt-trade-table`, `bt-trades-store`, `bt-download-trades`, `bt-export-btn`, `risk-metrics-text`, `risk-return-dist`, `risk-rolling-chart`, `risk-var-waterfall`
- Callback functions (3):
  - `run_backtest` (line 353): outputs [bt-trades-store.data, bt-returns-store.data, bt-total-return.children, bt-ann-return.children, bt-sharpe.children, bt-sortino.children, bt-max-dd.children, bt-win-rate.children, bt-profit-factor.children, bt-total-trades.children, bt-equity-curve.figure, bt-drawdown-chart.figure]; inputs [bt-run-btn.n_clicks]; state [bt-holding-period.value, bt-max-positions.value, bt-entry-threshold.value, bt-position-size.value, bt-risk-mgmt.value]
  - `update_risk_analytics` (line 554): outputs [risk-var-waterfall.figure, risk-metrics-text.children, risk-rolling-chart.figure, risk-return-dist.figure]; inputs [bt-returns-store.data]
  - `update_trade_analysis` (line 712): outputs [bt-trade-table.data, bt-regime-perf.figure]; inputs [bt-trades-store.data]
  - `export_trades_csv` (line ~730): outputs [bt-download-trades.data]; inputs [bt-export-btn.n_clicks]; state [bt-trades-store.data]

### `IV Surface`

- Route: `/iv-surface`
- Order: `6`
- File: `dash_ui/pages/iv_surface.py`
- Purpose: IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling.
- Declared UI IDs (15): `arb-build-btn`, `arb-div`, `arb-max-iter`, `arb-noise`, `arb-rate`, `arb-smiles-2d`, `arb-spot`, `arb-surface-3d`, `heston-compute-btn`, `heston-preset-dd`, `heston-smiles-2d`, `heston-surface-3d`, `svi-preset-dd`, `svi-smiles-2d`, `svi-surface-3d`
- Callback functions (5):
  - `set_svi_preset` (line 369): outputs [svi-a.value, svi-b.value, svi-rho.value, svi-m.value, svi-sigma.value]; inputs [svi-preset-dd.value]
  - `update_svi_surface` (line 388): outputs [svi-surface-3d.figure, svi-smiles-2d.figure]; inputs [svi-a.value, svi-b.value, svi-rho.value, svi-m.value, svi-sigma.value]
  - `set_heston_preset` (line 422): outputs [heston-v0.value, heston-theta.value, heston-kappa.value, heston-sigma.value, heston-rho.value]; inputs [heston-preset-dd.value]
  - `compute_heston_surface` (line 443): outputs [heston-surface-3d.figure, heston-smiles-2d.figure]; inputs [heston-compute-btn.n_clicks]; state [heston-v0.value, heston-theta.value, heston-kappa.value, heston-sigma.value, heston-rho.value]
  - `build_arb_free_surface` (line 562): outputs [arb-surface-3d.figure, arb-smiles-2d.figure]; inputs [arb-build-btn.n_clicks]; state [arb-spot.value, arb-rate.value, arb-div.value, arb-noise.value, arb-max-iter.value]

### `S&P Comparison`

- Route: `/sp-comparison`
- Order: `7`
- File: `dash_ui/pages/sp_comparison.py`
- Purpose: S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation.
- Declared UI IDs (11): `animation-interval`, `sp-alpha-beta-chart`, `sp-animate-btn`, `sp-corr-chart`, `sp-data-store`, `sp-dd-chart`, `sp-equity-chart`, `sp-frame-store`, `sp-load-btn`, `sp-period-dd`, `sp-relative-chart`
- Callback functions (3):
  - `load_and_compare` (line 161): outputs [sp-data-store.data, sp-strat-return.children, sp-bench-return.children, sp-alpha.children, sp-beta.children, sp-correlation.children, sp-tracking-error.children, sp-info-ratio.children, sp-equity-chart.figure, sp-corr-chart.figure, sp-alpha-beta-chart.figure, sp-relative-chart.figure, sp-dd-chart.figure, sp-frame-store.data]; inputs [sp-load-btn.n_clicks]; state [sp-period-dd.value]
  - `toggle_animation` (line 433): outputs [animation-interval.disabled, sp-animate-btn.children]; inputs [sp-animate-btn.n_clicks]; state [animation-interval.disabled]
  - `animate_equity_chart` (line 452): outputs [sp-equity-chart.figure, sp-frame-store.data]; inputs [animation-interval.n_intervals]; state [sp-data-store.data, sp-frame-store.data]

### `Autopilot & Events`

- Route: `/autopilot`
- Order: `8`
- File: `dash_ui/pages/autopilot_kalshi.py`
- Purpose: Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets.
- Declared UI IDs (14): `ap-discover-btn`, `ap-feature-mode-badge`, `ap-funnel-chart`, `ap-registry-btn`, `ap-strategy-table`, `kalshi-disagree-chart`, `kalshi-event-dd`, `kalshi-load-btn`, `kalshi-prob-chart`, `kalshi-timeline-chart`, `kalshi-wf-chart`, `paper-equity-chart`, `paper-load-btn`, `paper-positions-table`
- Callback functions (3):
  - `update_autopilot` (line 424): outputs [ap-strategy-table.data, ap-funnel-chart.figure, ap-feature-mode-badge.children]; inputs [ap-discover-btn.n_clicks, ap-registry-btn.n_clicks]
  - `load_paper_trading` (line 520): outputs [paper-equity-chart.figure, paper-positions-table.data]; inputs [paper-load-btn.n_clicks]
  - `load_kalshi_events` (line 596): outputs [kalshi-prob-chart.figure, kalshi-timeline-chart.figure, kalshi-wf-chart.figure, kalshi-disagree-chart.figure]; inputs [kalshi-load-btn.n_clicks]; state [kalshi-event-dd.value]

### `System Logs`

- Route: `/system-logs`
- Order: `2`
- File: `dash_ui/pages/system_logs.py`
- Purpose: System Logs -- Live log viewer with circular buffer, level filtering, and auto-refresh.
- Declared UI IDs (5): `log-interval`, `log-level-filter`, `log-clear-btn`, `log-table`, `log-container`
- Callback functions (2):
  - `update_log_table` (line ~80): outputs [log-table.data]; inputs [log-interval.n_intervals, log-level-filter.value]
  - `clear_logs` (line ~100): outputs [log-table.data]; inputs [log-clear-btn.n_clicks]

## Theme and Styling Utilities

### `theme.py`

- Module: `dash_ui/theme.py`
- Purpose: Theme constants and figure styling helpers for the Dash UI
- Top-level functions: `apply_plotly_template`, `create_figure`, `empty_figure`, `format_pct`, `format_number`, `metric_color`, `enhance_time_series`
