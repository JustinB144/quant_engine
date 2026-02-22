"""
Regime state badge component for displaying market regime indicators.

Provides a small colored badge showing the current market regime with
appropriate color coding and text labels.
"""
from dash import html

from ..theme import REGIME_COLORS, REGIME_NAMES, TEXT_PRIMARY


def regime_badge(regime_code: int) -> html.Div:
    """
    Create a colored badge displaying the regime name.

    Args:
        regime_code: Integer regime code (0, 1, 2, 3, ...).

    Returns:
        html.Div: Badge component with colored background and text.

    Example:
        regime_badge(0)  # Returns green "Trending Bull" badge
        regime_badge(1)  # Returns red "Trending Bear" badge
    """
    regime_name = REGIME_NAMES.get(regime_code, f"Regime {regime_code}")
    regime_color = REGIME_COLORS.get(regime_code, "#58a6ff")

    return html.Div(
        [
            html.Span(
                "●",
                style={
                    "color": regime_color,
                    "fontSize": "12px",
                    "marginRight": "6px",
                },
            ),
            html.Span(
                regime_name,
                style={
                    "fontSize": "12px",
                    "fontWeight": "500",
                    "color": TEXT_PRIMARY,
                    "fontFamily": "Menlo, monospace",
                },
            ),
        ],
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "padding": "4px 10px",
            "backgroundColor": f"{regime_color}20",  # 20% opacity
            "borderLeft": f"3px solid {regime_color}",
            "borderRadius": "3px",
            "whiteSpace": "nowrap",
        },
    )


__all__ = ["regime_badge"]
