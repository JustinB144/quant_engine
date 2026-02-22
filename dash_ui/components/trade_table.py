"""
Styled DataTable component for displaying trades with conditional formatting.

Features:
  - Dark theme with custom styling
  - Sortable columns, filter row, pagination
  - Conditional formatting: green for wins, red for losses
  - Fixed header row, native CSV export
  - Proportional column widths based on content
"""
from typing import List, Dict, Optional, Any

from dash import dash_table, html, dcc

from ..theme import BG_PRIMARY, BG_SECONDARY, TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_GREEN, ACCENT_RED


def trade_table(
    id: str,
    columns: List[Dict[str, str]],
    data: Optional[List[Dict[str, Any]]] = None,
) -> html.Div:
    """
    Create a styled DataTable for displaying trade-level data.

    Args:
        id: Unique identifier for the table component.
        columns: List of column definitions, e.g.,
                 [{"name": "Entry Date", "id": "entry_date"},
                  {"name": "Symbol", "id": "symbol"},
                  ...]
        data: List of row dictionaries. If None, table starts empty.

    Returns:
        html.Div: Container with styled DataTable, export button, and controls.

    Example:
        table = trade_table(
            id="trades-table",
            columns=[
                {"name": "Entry", "id": "entry_date"},
                {"name": "Symbol", "id": "symbol"},
                {"name": "Return %", "id": "net_return"},
            ],
            data=trades_df.to_dict("records"),
        )
    """
    if data is None:
        data = []

    return html.Div(
        [
            # Title bar with export button
            html.Div(
                [
                    html.Span(
                        "Trades",
                        style={
                            "fontSize": "14px",
                            "fontWeight": "bold",
                            "color": TEXT_PRIMARY,
                        },
                    ),
                    html.Button(
                        "📥 Export CSV",
                        id=f"{id}-export-btn",
                        n_clicks=0,
                        style={
                            "marginLeft": "auto",
                            "padding": "6px 12px",
                            "fontSize": "11px",
                            "backgroundColor": "#161b22",
                            "color": "#58a6ff",
                            "border": "1px solid #30363d",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "12px",
                    "backgroundColor": BG_SECONDARY,
                    "borderBottom": "1px solid #30363d",
                },
            ),

            # The DataTable
            dash_table.DataTable(
                id=id,
                columns=columns,
                data=data,
                style_cell={
                    "backgroundColor": BG_PRIMARY,
                    "color": TEXT_PRIMARY,
                    "border": "1px solid #21262d",
                    "padding": "8px",
                    "fontSize": "12px",
                    "fontFamily": "Menlo, monospace",
                    "textAlign": "left",
                    "minWidth": "80px",
                },
                style_header={
                    "backgroundColor": BG_SECONDARY,
                    "color": TEXT_SECONDARY,
                    "fontWeight": "bold",
                    "border": "1px solid #30363d",
                    "padding": "10px",
                    "fontSize": "11px",
                    "textAlign": "left",
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 100,
                },
                style_filter_cell={
                    "backgroundColor": "#1c2028",
                    "color": TEXT_SECONDARY,
                    "border": "1px solid #30363d",
                    "padding": "4px",
                    "fontSize": "11px",
                },
                style_data_conditional=[
                    # Winning trades: green background for net_return > 0
                    {
                        "if": {
                            "column_id": "net_return",
                            "filter_query": "{net_return} > 0",
                        },
                        "backgroundColor": "rgba(63, 185, 80, 0.15)",
                        "color": ACCENT_GREEN,
                        "fontWeight": "bold",
                    },
                    # Losing trades: red background for net_return < 0
                    {
                        "if": {
                            "column_id": "net_return",
                            "filter_query": "{net_return} < 0",
                        },
                        "backgroundColor": "rgba(248, 81, 73, 0.15)",
                        "color": ACCENT_RED,
                        "fontWeight": "bold",
                    },
                    # Neutral returns
                    {
                        "if": {
                            "column_id": "net_return",
                            "filter_query": "{net_return} = 0",
                        },
                        "color": TEXT_SECONDARY,
                    },
                    # Highlight confidence column
                    {
                        "if": {"column_id": "confidence"},
                        "color": "#79c0ff",
                    },
                ],
                # Enable sorting
                sort_action="native",
                # Enable filtering
                filter_action="native",
                # Enable pagination (50 rows per page)
                page_action="native",
                page_current=0,
                page_size=50,
                # Sticky header
                fixed_rows={"headers": True},
                # Virtualization for large datasets
                virtualization=True,
                # Tooltip on hover
                tooltip_data=[
                    {
                        column["id"]: {
                            "value": str(value),
                            "type": "markdown",
                        }
                        for column in columns
                    }
                    for value in data
                ],
                tooltip_duration=None,
                # CSS custom properties for styling
                css=[{
                    "selector": ".dash-table-container",
                    "rule": f"height: 100%; overflow-y: auto; background-color: {BG_PRIMARY};",
                }],
            ),

            # Info bar at bottom
            html.Div(
                [
                    html.Span(
                        id=f"{id}-info",
                        children=f"Showing {len(data)} trades",
                        style={
                            "fontSize": "11px",
                            "color": TEXT_SECONDARY,
                            "fontFamily": "Menlo, monospace",
                        },
                    ),
                ],
                style={
                    "padding": "8px 12px",
                    "backgroundColor": BG_SECONDARY,
                    "borderTop": "1px solid #30363d",
                    "fontSize": "11px",
                    "color": TEXT_SECONDARY,
                },
            ),

            # Download link (for CSV export functionality)
            dcc.Download(id=f"{id}-download"),
        ],
        style={
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "backgroundColor": BG_PRIMARY,
            "borderRadius": "4px",
            "overflow": "hidden",
            "border": "1px solid #30363d",
        },
    )


__all__ = ["trade_table"]
