"""
Quant Engine Professional Dashboard — Dash Application Factory.

Bloomberg-inspired dark-themed web interface providing unified access to all
system components: data management, model training, backtesting, risk
analytics, IV surface modeling, and S&P 500 benchmark comparison.

This module:
  - Configures the Plotly template for consistent styling
  - Creates the Dash app with multi-page support
  - Defines the sidebar navigation layout
  - Registers status bar with live clock updates
  - Applies Bootstrap theming (DARKLY) for consistent component styling
"""
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State

from .theme import apply_plotly_template, BG_PRIMARY, BG_SIDEBAR, TEXT_PRIMARY
from .components.sidebar import create_sidebar


def create_app() -> dash.Dash:
    """
    Create and configure the Dash application with multi-page support.

    Returns:
        dash.Dash: Configured Dash application instance ready to run.
    """
    # Register the Bloomberg dark Plotly template globally
    apply_plotly_template()

    # Create Dash app with multi-page support
    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        external_stylesheets=[
            dbc.themes.DARKLY,  # Bootstrap dark theme
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
        ],
        suppress_callback_exceptions=True,
        update_title=None,  # Don't update browser title on each page load
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "theme-color", "content": "#0d1117"},
            {"name": "description", "content": "Professional Quant Trading Dashboard"},
        ],
    )

    # Main layout: sidebar + page container + live status indicator
    app.layout = html.Div(
        [
            # URL location for routing
            dcc.Location(id="url", refresh="callback-nav"),

            # Sidebar (fixed left, 240px width) — styles handled by CSS class
            create_sidebar(),

            # Main content area (offset by sidebar width)
            html.Div(
                [
                    # Page container for multi-page routing (flex-grow)
                    html.Div(
                        dash.page_container,
                        style={"flex": "1 1 auto"},
                    ),

                    # Live status bar at bottom
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "QUANT ENGINE",
                                        style={
                                            "fontSize": "10px",
                                            "color": "#9ca3af",
                                            "fontFamily": "Menlo, monospace",
                                            "letterSpacing": "0.5px",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        id="live-time",
                                        children="--:--:--",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#c9d1d9",
                                            "fontFamily": "Menlo, monospace",
                                        },
                                    ),
                                    html.Span(
                                        " LIVE",
                                        style={
                                            "color": "#3fb950",
                                            "fontSize": "9px",
                                            "marginLeft": "8px",
                                            "fontWeight": "600",
                                            "letterSpacing": "0.5px",
                                        },
                                    ),
                                    html.Span(
                                        " \u25CF",
                                        style={
                                            "color": "#3fb950",
                                            "fontSize": "8px",
                                            "marginLeft": "4px",
                                            "animation": "pulse 2s ease-in-out infinite",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                        ],
                        style={
                            "padding": "6px 16px",
                            "borderTop": "1px solid #21262d",
                            "backgroundColor": "#010409",
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "fontSize": "11px",
                            "flexShrink": "0",
                        },
                    ),
                ],
                className="main-content",
                style={
                    "marginLeft": "240px",
                    "display": "flex",
                    "flexDirection": "column",
                    "minHeight": "100vh",
                },
            ),

            # Interval for live time updates
            dcc.Interval(id="live-time-interval", interval=1000, n_intervals=0),
        ],
        style={
            "backgroundColor": BG_PRIMARY,
            "minHeight": "100vh",
        },
    )

    # Callback to update live time display
    @callback(
        Output("live-time", "children"),
        Input("live-time-interval", "n_intervals"),
    )
    def update_live_time(n_intervals: int) -> str:
        """Update the live time display in the status bar every second."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    return app
