"""Reusable health check display component."""
from dash import html

STATUS_ICONS = {"pass": "\u2713", "warn": "\u26A0", "fail": "\u2717", "info": "\u2139"}
STATUS_COLORS = {"pass": "#3fb950", "warn": "#d29922", "fail": "#f85149", "info": "#58a6ff"}


def health_check_item(label, status="pass", detail=None):
    """Render a single health check row with icon, label, and optional detail."""
    icon = STATUS_ICONS.get(status, "\u2022")
    color = STATUS_COLORS.get(status, "#8b949e")

    children = [
        html.Span(icon, style={"color": color, "marginRight": "8px", "fontSize": "14px"}),
        html.Span(label, style={"color": "#c9d1d9", "fontSize": "13px"}),
    ]
    if detail:
        children.append(html.Span(
            f"  {detail}",
            style={"color": "#8b949e", "fontSize": "11px", "fontFamily": "Menlo, monospace"},
        ))

    return html.Div(
        children,
        style={"padding": "6px 0", "borderBottom": "1px solid #21262d"},
    )


def health_check_list(checks):
    """Render a list of health checks.

    Args:
        checks: List of dicts with keys 'label', 'status', and optional 'detail'.
    """
    return html.Div(
        [health_check_item(**c) for c in checks],
        style={"padding": "8px 0"},
    )
