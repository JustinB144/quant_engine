"""
Dash page modules for the Quant Engine Dashboard.

Multi-page application structure using dash.register_page().
Each module in this package registers itself as a page with a URL route.

Pages (to be created):
  - dashboard.py: Main performance dashboard (/)
  - system_health.py: System health checks and diagnostics (/health)
  - data_explorer.py: Data inspection and cache management (/data)
  - model_lab.py: Model training, feature analysis, and performance (/model)
  - signal_desk.py: Signal generation, testing, and statistics (/signals)
  - backtest_risk.py: Backtest results visualization and risk metrics (/backtest)
  - iv_surface.py: Implied volatility surface modeling and analysis (/iv-surface)
  - sp_compare.py: S&P 500 benchmark comparison and attribution (/sp-compare)
  - autopilot.py: Strategy discovery, promotion funnel, and paper trading (/autopilot)

Each page module should:
  1. Call dash.register_page(__name__, ...)
  2. Define a layout function or html.Div
  3. Include any necessary callbacks for interactivity
  4. Import components from dash_ui.components
  5. Use the Bloomberg dark theme consistently
"""

__all__ = []
