"""
Alert banner component for displaying system messages and warnings.

Provides styled alert banners with different severity levels:
  - "success": Green for positive outcomes
  - "info": Blue for informational messages
  - "warning": Amber for warnings
  - "danger": Red for errors and critical issues
"""
from typing import Literal

import dash_bootstrap_components as dbc
from dash import html

from ..theme import (
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    ACCENT_GREEN,
    ACCENT_BLUE,
    ACCENT_AMBER,
    ACCENT_RED,
    BG_TERTIARY,
)


def alert_banner(
    message: str,
    severity: Literal["success", "info", "warning", "danger", "error"] = "info",
    icon: str = "ℹ",
    dismissable: bool = False,
) -> dbc.Alert:
    """
    Create a styled alert banner message.

    Args:
        message: Alert message text (can include HTML).
        severity: Alert level - one of:
                  - "success" (green, ✓)
                  - "info" (blue, ℹ)
                  - "warning" (amber, ⚠)
                  - "danger" or "error" (red, ✗)
        icon: Icon character to display (default matches severity).
        dismissable: If True, alert can be dismissed by user.

    Returns:
        dbc.Alert: Styled alert component.

    Example:
        alert_banner("Model training completed successfully!", severity="success")
        alert_banner("Cache is stale (>7 days old)", severity="warning")
        alert_banner("Critical error: insufficient data", severity="danger")
    """
    # Map severity to colors and default icons
    severity_config = {
        "success": {
            "color": "success",
            "bg_color": ACCENT_GREEN,
            "icon": "✓",
            "border_color": ACCENT_GREEN,
        },
        "info": {
            "color": "info",
            "bg_color": ACCENT_BLUE,
            "icon": "ℹ",
            "border_color": ACCENT_BLUE,
        },
        "warning": {
            "color": "warning",
            "bg_color": ACCENT_AMBER,
            "icon": "⚠",
            "border_color": ACCENT_AMBER,
        },
        "danger": {
            "color": "danger",
            "bg_color": ACCENT_RED,
            "icon": "✗",
            "border_color": ACCENT_RED,
        },
        "error": {  # Alias for danger
            "color": "danger",
            "bg_color": ACCENT_RED,
            "icon": "✗",
            "border_color": ACCENT_RED,
        },
    }

    config = severity_config.get(severity, severity_config["info"])
    bg_color = config["bg_color"]

    return dbc.Alert(
        [
            html.Div(
                [
                    html.Span(
                        icon,
                        style={
                            "display": "inline-block",
                            "marginRight": "8px",
                            "fontSize": "14px",
                            "fontWeight": "bold",
                            "color": bg_color,
                        },
                    ),
                    html.Span(
                        message,
                        style={
                            "fontSize": "13px",
                            "color": TEXT_PRIMARY,
                            "lineHeight": "1.4",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start"},
            ),
        ],
        style={
            "backgroundColor": f"{bg_color}15",
            "borderLeft": f"4px solid {bg_color}",
            "borderRadius": "3px",
            "padding": "10px 12px",
            "marginBottom": "12px",
            "border": f"1px solid {bg_color}30",
            "color": "#ffffff",
        },
        dismissable=dismissable,
        is_open=True,
    )


__all__ = ["alert_banner"]
