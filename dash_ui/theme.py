"""
Bloomberg-inspired dark theme for the Quant Engine Dash Dashboard.

Ports the exact color palette from the tkinter theme and creates
a custom Plotly template applied globally to all charts.
"""
import plotly.graph_objects as go
import plotly.io as pio

# ── Color Palette (exact port from ui/theme.py) ───────────────────────
BG_PRIMARY = "#0d1117"
BG_SECONDARY = "#161b22"
BG_TERTIARY = "#1c2028"
BG_SIDEBAR = "#010409"
BG_HOVER = "#1f2937"
BG_ACTIVE = "#1a2332"
BG_INPUT = "#0d1117"

BORDER = "#30363d"
BORDER_LIGHT = "#21262d"
BORDER_FOCUS = "#58a6ff"

TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c9d1d9"
TEXT_TERTIARY = "#8b949e"
TEXT_HEADING = "#ffffff"

ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_AMBER = "#d29922"
ACCENT_PURPLE = "#bc8cff"
ACCENT_CYAN = "#39d2c0"
ACCENT_ORANGE = "#f0883e"

CHART_COLORS = [
    "#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff",
    "#39d2c0", "#f0883e", "#79c0ff", "#56d364", "#ff7b72",
    "#e3b341", "#d2a8ff", "#76e4cc", "#ffa657",
]

REGIME_COLORS = {
    0: ACCENT_GREEN,    # Trending Bull
    1: ACCENT_RED,      # Trending Bear
    2: ACCENT_BLUE,     # Mean Reverting
    3: ACCENT_AMBER,    # High Volatility
}
REGIME_NAMES = {
    0: "Trending Bull",
    1: "Trending Bear",
    2: "Mean Reverting",
    3: "High Volatility",
}

# Status colors
STATUS_COLORS = {
    "pass": ACCENT_GREEN,
    "warn": ACCENT_AMBER,
    "fail": ACCENT_RED,
    "info": ACCENT_BLUE,
}


def apply_plotly_template():
    """Register and activate the Bloomberg dark Plotly template."""
    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=BG_PRIMARY,
            plot_bgcolor=BG_SECONDARY,
            font=dict(family="Menlo, monospace", color=TEXT_PRIMARY, size=12),
            title=dict(font=dict(size=14, color=TEXT_PRIMARY)),
            xaxis=dict(
                gridcolor=BORDER_LIGHT,
                gridwidth=0.5,
                zerolinecolor=BORDER,
                linecolor=BORDER,
                tickfont=dict(color=TEXT_SECONDARY, size=10),
                title_font=dict(color=TEXT_SECONDARY, size=11),
            ),
            yaxis=dict(
                gridcolor=BORDER_LIGHT,
                gridwidth=0.5,
                zerolinecolor=BORDER,
                linecolor=BORDER,
                tickfont=dict(color=TEXT_SECONDARY, size=10),
                title_font=dict(color=TEXT_SECONDARY, size=11),
            ),
            legend=dict(
                bgcolor=BG_SECONDARY,
                bordercolor=BORDER,
                borderwidth=1,
                font=dict(color=TEXT_PRIMARY, size=10),
            ),
            colorway=CHART_COLORS,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor=BG_TERTIARY,
                font_color=TEXT_PRIMARY,
                bordercolor=BORDER,
            ),
            margin=dict(l=50, r=20, t=40, b=40),
        )
    )
    pio.templates["bloomberg_dark"] = template
    pio.templates.default = "bloomberg_dark"


def create_figure(**kwargs) -> go.Figure:
    """
    Create a Plotly figure with the Bloomberg dark template applied.

    Returns a go.Figure configured with tight margins and the custom theme.
    Accepts any kwargs to update_layout.

    Returns:
        go.Figure: Figure with template applied and tight margins.
    """
    fig = go.Figure()
    margins = kwargs.pop("margin", dict(l=40, r=20, t=30, b=30))
    fig.update_layout(
        template="bloomberg_dark",
        margin=margins,
        hovermode="x unified",
        **kwargs
    )
    return fig


def empty_figure(message: str = "No data available") -> go.Figure:
    """
    Return an empty Plotly figure with a centered message.

    Args:
        message: Text message to display in center of figure.

    Returns:
        go.Figure: Empty figure with centered message annotation.
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=TEXT_TERTIARY),
        )],
        height=300,
    )
    return fig


def enhance_time_series(fig: go.Figure) -> go.Figure:
    """
    Apply range selector buttons and crosshair cursor to a time series figure.

    Adds 1M/3M/6M/1Y/ALL range buttons and a vertical spike crosshair
    for professional chart interactivity.

    Args:
        fig: Plotly figure with time-series x-axis.

    Returns:
        go.Figure: Enhanced figure (mutated in place).
    """
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(label="ALL", step="all"),
            ],
            bgcolor=BG_TERTIARY,
            activecolor=ACCENT_BLUE,
            bordercolor=BORDER,
            font=dict(color=TEXT_SECONDARY, size=10),
            x=0,
            y=1.08,
        ),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=BORDER,
        spikedash="dot",
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=BORDER,
        spikedash="dot",
    )
    fig.update_layout(
        spikedistance=-1,
        hovermode="x unified",
    )
    return fig


def format_pct(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as percentage string.

    Args:
        value: Decimal value (e.g., 0.1234 for 12.34%).
        decimals: Number of decimal places (default: 1).

    Returns:
        str: Formatted percentage string (e.g., "12.3%").
    """
    if not isinstance(value, (int, float)) or not (value == value):  # NaN check
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with thousand separators and decimal places.

    Args:
        value: Numeric value to format.
        decimals: Number of decimal places (default: 2).

    Returns:
        str: Formatted number string with thousand separators.
    """
    if not isinstance(value, (int, float)) or not (value == value):  # NaN check
        return "N/A"
    return f"{value:,.{decimals}f}"


def metric_color(value: float, positive_is_good: bool = True) -> str:
    """
    Return color hex code based on value sign and preference.

    Args:
        value: Numeric value to evaluate.
        positive_is_good: If True, positive values get green. If False, negative values get green.

    Returns:
        str: Hex color code ("#3fb950" green or "#f85149" red).
    """
    if not isinstance(value, (int, float)) or not (value == value):  # NaN check
        return TEXT_SECONDARY

    is_positive = value >= 0
    if positive_is_good:
        return ACCENT_GREEN if is_positive else ACCENT_RED
    else:
        return ACCENT_RED if is_positive else ACCENT_GREEN
