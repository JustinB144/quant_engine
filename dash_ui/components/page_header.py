"""Reusable page header component with title, subtitle, and action buttons."""
from dash import html

from ..theme import BORDER, TEXT_PRIMARY, TEXT_TERTIARY


def create_page_header(title: str, subtitle: str = "", actions=None):
    """
    Create a consistent page header with title, subtitle, and optional actions.

    Args:
        title: Page title text.
        subtitle: Optional subtitle or description.
        actions: Optional list of Dash components (buttons, badges) for the right side.

    Returns:
        html.Div: Styled page header component.
    """
    left = [
        html.H2(
            title,
            className="page-title",
            style={
                "display": "inline-block",
                "borderBottom": "none",
                "marginBottom": "0",
                "paddingBottom": "0",
            },
        ),
    ]
    if subtitle:
        left.append(
            html.Span(
                subtitle,
                style={
                    "fontSize": "12px",
                    "color": TEXT_TERTIARY,
                    "marginLeft": "12px",
                    "verticalAlign": "middle",
                },
            )
        )

    right = []
    if actions:
        right = actions if isinstance(actions, list) else [actions]

    return html.Div(
        [
            html.Div(left, style={"display": "flex", "alignItems": "baseline"}),
            html.Div(right, style={"display": "flex", "gap": "8px", "alignItems": "center"}),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "borderBottom": f"1px solid {BORDER}",
            "paddingBottom": "12px",
            "marginBottom": "20px",
        },
    )


__all__ = ["create_page_header"]
