# `dash_ui` Package Guide

## Purpose

Active Dash web UI for system observability, analysis workflows, and interactive diagnostics.

## Package Summary

- Modules: 28
- Classes: 3
- Top-level functions: 128
- LOC: 9,726

## How This Package Fits Into the System

- Reads artifacts/results/state from core subsystems
- Should not own core trading logic
- Provides page-specific workflows for humans

## Module Index

| Module | Lines | Classes | Top-level Functions | Module Intent |
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
| `dash_ui/components/page_header.py` | 35 | 0 | 1 | Reusable page header component with title, subtitle, and action buttons. |
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
| `dash_ui/pages/system_logs.py` | 120 | 1 | 3 | System Logs -- Live log viewer with circular buffer and level filtering. |
| `dash_ui/pages/system_health.py` | 1081 | 0 | 14 | System Health Console -- comprehensive health assessment for the Quant Engine. |
| `dash_ui/server.py` | 52 | 0 | 0 | Entry point for the Quant Engine Dash application. |
| `dash_ui/theme.py` | 205 | 0 | 6 | Bloomberg-inspired dark theme for the Quant Engine Dash Dashboard. |

## Module Details

### `dash_ui/__init__.py`
- Intent: Quantitative Trading Engine - Dash UI Package
- Classes: none
- Top-level functions: none

### `dash_ui/app.py`
- Intent: Quant Engine Professional Dashboard — Dash Application Factory.
- Classes: none
- Top-level functions: `create_app`

### `dash_ui/components/__init__.py`
- Intent: Reusable Dash components for the Quant Engine Dashboard.
- Classes: none
- Top-level functions: none

### `dash_ui/components/alert_banner.py`
- Intent: Alert banner component for displaying system messages and warnings.
- Classes: none
- Top-level functions: `alert_banner`

### `dash_ui/components/chart_utils.py`
- Intent: Plotly chart factory functions for the Quant Engine Dashboard.
- Classes: none
- Top-level functions: `line_chart`, `area_chart`, `bar_chart`, `heatmap_chart`, `surface_3d`, `equity_curve`, `regime_timeline`, `dual_axis_chart`, `candlestick_chart`, `scatter_chart`, `radar_chart`, `histogram_chart`

### `dash_ui/components/health_check_list.py`
- Intent: Reusable health check display component.
- Classes: none
- Top-level functions: `health_check_item`, `health_check_list`

### `dash_ui/components/metric_card.py`
- Intent: Reusable KPI metric card component.
- Classes: none
- Top-level functions: `metric_card`

### `dash_ui/components/regime_badge.py`
- Intent: Regime state badge component for displaying market regime indicators.
- Classes: none
- Top-level functions: `regime_badge`

### `dash_ui/components/sidebar.py`
- Intent: Sidebar navigation component with active-state highlighting.
- Classes: none
- Top-level functions: `_nav_item`, `create_sidebar`, `update_active_nav`

### `dash_ui/components/page_header.py`
- Intent: Reusable page header component with title, subtitle, and action buttons.
- Classes: none
- Top-level functions: `create_page_header`

### `dash_ui/components/status_bar.py`
- Intent: Bottom status bar component.
- Classes: none
- Top-level functions: `create_status_bar`

### `dash_ui/components/trade_table.py`
- Intent: Styled DataTable component for displaying trades with conditional formatting.
- Classes: none
- Top-level functions: `trade_table`

### `dash_ui/data/__init__.py`
- Intent: Data loading and caching layer for the Dash UI.
- Classes: none
- Top-level functions: none

### `dash_ui/data/cache.py`
- Intent: Caching layer for Dash UI data loading operations.
- Classes: none
- Top-level functions: `init_cache`, `cached`

### `dash_ui/data/loaders.py`
- Intent: Data loading and computation functions for the Dash UI.
- Classes:
  - `HealthCheck`: Single health check result.
  - `SystemHealthPayload`: Full system health assessment.
- Top-level functions: `load_trades`, `build_portfolio_returns`, `_read_close_returns`, `load_benchmark_returns`, `load_factor_proxies`, `compute_risk_metrics`, `compute_regime_payload`, `compute_model_health`, `load_feature_importance`, `compute_health_scores`, `score_to_status`, `collect_health_data`, `_check_data_integrity`, `_check_promotion_contract`, `_check_walkforward`, `_check_execution`, `_check_complexity`, `_check_strengths`

### `dash_ui/pages/__init__.py`
- Intent: Dash page modules for the Quant Engine Dashboard.
- Classes: none
- Top-level functions: none

### `dash_ui/pages/autopilot_kalshi.py`
- Intent: Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets.
- Classes: none
- Top-level functions: `_demo_strategy_candidates`, `_demo_promotion_funnel`, `_demo_paper_equity`, `_demo_paper_positions`, `_demo_kalshi_events`, `update_autopilot`, `load_paper_trading`, `load_kalshi_events`

### `dash_ui/pages/backtest_risk.py`
- Intent: Backtest & Risk -- Equity curves, risk metrics, and trade analysis.
- Classes: none
- Top-level functions: `_control_group`, `_metrics_row`, `run_backtest`, `update_risk_analytics`, `update_trade_analysis`

### `dash_ui/pages/dashboard.py`
- Intent: Dashboard -- Portfolio Intelligence Overview.
- Classes: none
- Top-level functions: `_card_panel`, `_pct`, `_fmt`, `load_dashboard_data`, `update_metric_cards`, `render_tab_content`, `_render_portfolio_tab`, `_render_regime_tab`, `_render_model_tab`, `_render_features_tab`, `_render_trades_tab`, `_render_risk_tab`

### `dash_ui/pages/data_explorer.py`
- Intent: Data Explorer -- OHLCV visualization and quality analysis.
- Classes: none
- Top-level functions: `_download_ticker`, `_generate_demo_data`, `_compute_sma`, `_df_to_store`, `_store_to_df`, `load_data`, `render_ticker_list`, `select_ticker`, `update_price_chart`, `update_volume_chart`, `update_stats_bar`, `toggle_quality_modal`

### `dash_ui/pages/iv_surface.py`
- Intent: IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling.
- Classes: none
- Top-level functions: `compute_svi_surface`, `compute_svi_smiles`, `_svi_slider`, `_build_surface_figure`, `_build_smiles_figure`, `set_svi_preset`, `update_svi_surface`, `set_heston_preset`, `compute_heston_surface`, `build_arb_free_surface`

### `dash_ui/pages/model_lab.py`
- Intent: Model Lab -- Feature engineering, regime detection, and model training.
- Classes: none
- Top-level functions: `_demo_feature_importance`, `load_features`, `_demo_regime_payload`, `run_regime_detection`, `train_model`, `_build_cv_chart`, `_build_demo_cv_chart`

### `dash_ui/pages/signal_desk.py`
- Intent: Signal Desk -- Prediction generation and signal ranking.
- Classes: none
- Top-level functions: `_generate_demo_signals`, `_try_live_signals`, `generate_signals`

### `dash_ui/pages/sp_comparison.py`
- Intent: S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation.
- Classes: none
- Top-level functions: `load_and_compare`, `toggle_animation`, `animate_equity_chart`

### `dash_ui/pages/system_logs.py`
- Intent: System Logs -- Live log viewer with circular buffer and level filtering.
- Classes:
  - `DashLogHandler`: Logging handler that captures records into a circular buffer for the UI.
    - Methods: `emit`
- Top-level functions: `install_log_handler`, `update_log_table`, `clear_logs`

### `dash_ui/pages/system_health.py`
- Intent: System Health Console -- comprehensive health assessment for the Quant Engine.
- Classes: none
- Top-level functions: `_card_panel`, `_status_span`, `_check_row`, `_instruction_banner`, `_score_color`, `load_health_data`, `update_health_cards`, `render_health_tab`, `_render_overview_tab`, `_render_data_tab`, `_render_promotion_tab`, `_render_wf_tab`, `_render_execution_tab`, `_render_complexity_tab`

### `dash_ui/server.py`
- Intent: Entry point for the Quant Engine Dash application.
- Classes: none
- Top-level functions: none

### `dash_ui/theme.py`
- Intent: Bloomberg-inspired dark theme for the Quant Engine Dash Dashboard.
- Classes: none
- Top-level functions: `apply_plotly_template`, `create_figure`, `empty_figure`, `format_pct`, `format_number`, `metric_color`, `enhance_time_series`



## Related Docs

- `../docs/reports/QUANT_ENGINE_SYSTEM_INTENT_COMPONENT_AUDIT.md` (deep system audit)
- `../docs/reference/SOURCE_API_REFERENCE.md` (full API inventory)
- `../docs/architecture/SYSTEM_ARCHITECTURE_AND_FLOWS.md` (subsystem interactions)
