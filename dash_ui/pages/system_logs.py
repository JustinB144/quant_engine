"""System Logs -- Real-time log viewer for the Quant Engine."""
import logging
from collections import deque
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html

from quant_engine.dash_ui.components.page_header import create_page_header
from quant_engine.dash_ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_PRIMARY,
    BG_SECONDARY,
    BORDER,
    BORDER_LIGHT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)

dash.register_page(__name__, path="/system-logs", name="System Logs", order=2)

# ---------------------------------------------------------------------------
# Circular buffer log handler
# ---------------------------------------------------------------------------

_LOG_BUFFER: deque = deque(maxlen=500)

LEVEL_COLORS = {
    "DEBUG": TEXT_TERTIARY,
    "INFO": ACCENT_BLUE,
    "WARNING": ACCENT_AMBER,
    "ERROR": ACCENT_RED,
    "CRITICAL": ACCENT_RED,
}


class DashLogHandler(logging.Handler):
    """Logging handler that writes records to a shared circular buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        """emit."""
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "level": record.levelname,
                "module": record.name,
                "message": self.format(record),
            }
            _LOG_BUFFER.appendleft(entry)
        except (ValueError, KeyError, TypeError):
            pass


def install_log_handler() -> None:
    """Install the DashLogHandler on the quant_engine root logger."""
    handler = DashLogHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger("quant_engine")
    # Avoid duplicate handlers on reload
    for h in root_logger.handlers[:]:
        if isinstance(h, DashLogHandler):
            root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)


# Install handler on module import
install_log_handler()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        dcc.Interval(id="logs-interval", interval=3000, n_intervals=0),
        create_page_header(
            "System Logs",
            subtitle="Real-time engine log viewer",
            actions=[
                dbc.Button(
                    "Clear Logs",
                    id="logs-clear-btn",
                    className="btn-secondary",
                    size="sm",
                    style={"fontSize": "11px"},
                ),
            ],
        ),
        # Log level filter
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Filter Level",
                            style={
                                "fontSize": "11px",
                                "color": TEXT_TERTIARY,
                                "marginBottom": "4px",
                                "fontFamily": "Menlo, monospace",
                            },
                        ),
                        dcc.Dropdown(
                            id="logs-level-filter",
                            options=[
                                {"label": "All", "value": "ALL"},
                                {"label": "DEBUG", "value": "DEBUG"},
                                {"label": "INFO", "value": "INFO"},
                                {"label": "WARNING", "value": "WARNING"},
                                {"label": "ERROR", "value": "ERROR"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"fontSize": "12px"},
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    html.Div(
                        id="logs-count-badge",
                        style={
                            "fontSize": "11px",
                            "color": TEXT_TERTIARY,
                            "fontFamily": "Menlo, monospace",
                            "paddingTop": "24px",
                        },
                    ),
                    md=4,
                ),
            ],
            className="mb-3",
        ),
        # Log table
        dash_table.DataTable(
            id="logs-table",
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Level", "id": "level"},
                {"name": "Module", "id": "module"},
                {"name": "Message", "id": "message"},
            ],
            data=[],
            page_size=50,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1c2028",
                "color": TEXT_PRIMARY,
                "fontWeight": "600",
                "fontSize": "11px",
                "borderBottom": f"2px solid {BORDER}",
                "fontFamily": "Menlo, monospace",
                "textAlign": "left",
                "padding": "8px 10px",
            },
            style_cell={
                "backgroundColor": BG_SECONDARY,
                "color": TEXT_SECONDARY,
                "fontSize": "11px",
                "fontFamily": "Menlo, monospace",
                "border": f"1px solid {BORDER_LIGHT}",
                "textAlign": "left",
                "padding": "6px 10px",
                "maxWidth": "600px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_cell_conditional=[
                {"if": {"column_id": "timestamp"}, "width": "140px"},
                {"if": {"column_id": "level"}, "width": "80px"},
                {"if": {"column_id": "module"}, "width": "180px"},
            ],
            style_data_conditional=[
                {
                    "if": {"filter_query": '{level} = "ERROR"'},
                    "color": ACCENT_RED,
                },
                {
                    "if": {"filter_query": '{level} = "CRITICAL"'},
                    "color": ACCENT_RED,
                    "fontWeight": "bold",
                },
                {
                    "if": {"filter_query": '{level} = "WARNING"'},
                    "color": ACCENT_AMBER,
                },
                {
                    "if": {"filter_query": '{level} = "INFO"'},
                    "color": ACCENT_BLUE,
                },
                {
                    "if": {"filter_query": '{level} = "DEBUG"'},
                    "color": TEXT_TERTIARY,
                },
                {
                    "if": {"state": "active"},
                    "backgroundColor": "#1f2937",
                    "border": f"1px solid {ACCENT_BLUE}",
                },
            ],
            style_filter={
                "backgroundColor": BG_PRIMARY,
                "color": TEXT_SECONDARY,
            },
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("logs-table", "data"),
    Output("logs-count-badge", "children"),
    Input("logs-interval", "n_intervals"),
    Input("logs-clear-btn", "n_clicks"),
    Input("logs-level-filter", "value"),
    prevent_initial_call=False,
)
def update_logs(n_intervals, clear_clicks, level_filter):
    """Refresh the log table from the circular buffer."""
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "logs-clear-btn.n_clicks":
        _LOG_BUFFER.clear()

    entries = list(_LOG_BUFFER)

    if level_filter and level_filter != "ALL":
        level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        min_level = level_order.get(level_filter, 0)
        entries = [e for e in entries if level_order.get(e["level"], 0) >= min_level]

    count_text = f"{len(entries)} entries  |  Buffer: {len(_LOG_BUFFER)}/500"

    return entries, count_text
