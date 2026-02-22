"""Reusable KPI metric card component."""
from dash import html


def metric_card(label, value, color="#58a6ff", subtitle=None, icon=None):
    """Create a styled KPI metric card.

    Args:
        label: Card title text.
        value: Main metric value (string).
        color: Accent color for the value text.
        subtitle: Optional smaller text below the value.
        icon: Optional icon character to display.
    """
    children = []

    header_items = []
    if icon:
        header_items.append(html.Span(
            icon,
            style={"fontSize": "14px", "color": "#9ca3af", "marginRight": "6px"},
        ))
    header_items.append(html.Span(
        label,
        style={"fontSize": "11px", "color": "#9ca3af", "textTransform": "uppercase",
               "letterSpacing": "0.5px"},
    ))
    children.append(html.Div(header_items, style={"marginBottom": "8px"}))

    children.append(html.Div(
        value,
        style={"fontSize": "22px", "fontWeight": "bold", "color": color,
               "fontFamily": "Menlo, monospace", "lineHeight": "1.2"},
    ))

    if subtitle:
        children.append(html.Div(
            subtitle,
            style={"fontSize": "10px", "color": "#9ca3af", "marginTop": "4px",
                   "fontFamily": "Menlo, monospace"},
        ))

    return html.Div(
        children,
        className="metric-card",
    )
