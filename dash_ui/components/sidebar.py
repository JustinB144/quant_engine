"""Sidebar navigation component with active-state highlighting."""
from dash import html, dcc, callback, Input, Output, ALL, ctx
import dash

NAV_ITEMS = [
    {"key": "dashboard",   "label": "Dashboard",          "icon": "fa-solid fa-chart-line",       "path": "/"},
    {"key": "health",      "label": "System Health",      "icon": "fa-solid fa-heart-pulse",      "path": "/system-health"},
    {"key": "logs",        "label": "System Logs",        "icon": "fa-solid fa-terminal",         "path": "/system-logs"},
    {"key": "data",        "label": "Data Explorer",      "icon": "fa-solid fa-database",         "path": "/data-explorer"},
    {"key": "model",       "label": "Model Lab",          "icon": "fa-solid fa-flask",            "path": "/model-lab"},
    {"key": "signals",     "label": "Signal Desk",        "icon": "fa-solid fa-signal",           "path": "/signal-desk"},
    {"key": "backtest",    "label": "Backtest & Risk",    "icon": "fa-solid fa-chart-area",       "path": "/backtest-risk"},
    {"key": "iv_surface",  "label": "IV Surface",         "icon": "fa-solid fa-layer-group",      "path": "/iv-surface"},
    {"key": "sp_compare",  "label": "S&P Comparison",     "icon": "fa-solid fa-scale-balanced",   "path": "/sp-comparison"},
    {"key": "autopilot",   "label": "Autopilot & Events", "icon": "fa-solid fa-robot",            "path": "/autopilot"},
]


def _nav_item(item):
    """Render a single navigation item."""
    return dcc.Link(
        html.Div(
            [
                html.I(
                    className=item["icon"],
                    style={"fontSize": "13px", "marginRight": "10px", "width": "18px",
                           "display": "inline-block", "textAlign": "center",
                           "opacity": "0.85"},
                ),
                html.Span(item["label"], style={"fontSize": "13px"}),
            ],
            id={"type": "nav-item", "key": item["key"]},
            className="nav-item",
        ),
        href=item["path"],
        style={"textDecoration": "none"},
    )


def create_sidebar():
    """Create the full sidebar layout."""
    return html.Div(
        [
            # Logo section
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fa-solid fa-bolt",
                                   style={"color": "#58a6ff", "marginRight": "8px",
                                          "fontSize": "16px"}),
                            html.Span(
                                "QUANT ENGINE",
                                style={"fontSize": "15px", "fontWeight": "700",
                                       "color": "#58a6ff", "letterSpacing": "1.5px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        "Professional Trading System",
                        style={"fontSize": "10px", "color": "#8b949e",
                               "marginTop": "4px", "letterSpacing": "0.3px"},
                    ),
                ],
                style={"padding": "20px 16px 8px"},
            ),
            # Separator
            html.Hr(style={"borderColor": "#30363d", "margin": "12px 16px 16px"}),
            # Section label
            html.Div("NAVIGATION", style={
                "fontSize": "9px", "color": "#8b949e", "letterSpacing": "1.5px",
                "fontWeight": "600", "padding": "0 16px 8px",
                "fontFamily": "Menlo, monospace",
            }),
            # Navigation items
            html.Div([_nav_item(item) for item in NAV_ITEMS]),
            # Bottom section
            html.Div(
                [
                    html.Hr(style={"borderColor": "#30363d", "margin": "0 0 12px"}),
                    html.Div(
                        [
                            html.Span(
                                "v2.0.0",
                                style={"fontSize": "10px", "color": "#8b949e",
                                       "fontFamily": "Menlo, monospace"},
                            ),
                            html.Span(
                                " | LIVE",
                                style={"fontSize": "9px", "color": "#3fb950",
                                       "fontFamily": "Menlo, monospace",
                                       "fontWeight": "600"},
                            ),
                        ],
                    ),
                ],
                style={"position": "absolute", "bottom": "16px", "left": "16px",
                       "right": "16px"},
            ),
        ],
        className="sidebar",
    )


@callback(
    Output({"type": "nav-item", "key": ALL}, "className"),
    Input("url", "pathname"),
)
def update_active_nav(pathname):
    """Highlight the active navigation item based on URL."""
    if pathname is None:
        pathname = "/"
    classes = []
    for item in NAV_ITEMS:
        if pathname == item["path"] or (pathname == "/" and item["path"] == "/"):
            classes.append("nav-item active")
        elif pathname.startswith(item["path"]) and item["path"] != "/":
            classes.append("nav-item active")
        else:
            classes.append("nav-item")
    return classes
