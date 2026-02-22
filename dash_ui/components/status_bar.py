"""Bottom status bar component."""
import sys
from datetime import datetime
from dash import html


def create_status_bar():
    """Create the bottom status bar."""
    return html.Div(
        [
            html.Span(
                f"  Ready  |  Python {sys.version.split()[0]}  |  "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                style={"fontSize": "11px", "color": "#9ca3af",
                       "fontFamily": "Menlo, monospace"},
            ),
            html.Span(
                "Quant Engine Dashboard  ",
                style={"fontSize": "11px", "color": "#9ca3af",
                       "fontFamily": "Menlo, monospace", "float": "right"},
            ),
        ],
        style={
            "backgroundColor": "#1c2028",
            "padding": "6px 12px",
            "borderTop": "1px solid #30363d",
            "position": "fixed",
            "bottom": "0",
            "left": "240px",
            "right": "0",
            "zIndex": "100",
            "height": "28px",
        },
    )
