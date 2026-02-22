"""
Plotly chart factory functions for the Quant Engine Dashboard.

All functions return go.Figure instances with the Bloomberg dark template
pre-applied, consistent margins, and professional styling.

Functions provided:
  - line_chart: Multi-line chart with legend
  - area_chart: Filled area chart
  - bar_chart: Horizontal/vertical bar chart
  - heatmap_chart: Annotated heatmap (e.g., correlation matrix)
  - surface_3d: 3D surface plot
  - equity_curve: Equity curve with drawdown shading
  - regime_timeline: Colored regime state bands
  - dual_axis_chart: Two y-axes with different scales
  - candlestick_chart: OHLCV candlestick with volume subplot
  - scatter_chart: Scatter plot with optional color/size encoding
  - radar_chart: Polar/radar chart
  - histogram_chart: Distribution histogram
"""
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    create_figure,
    CHART_COLORS,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_AMBER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BG_SECONDARY,
    BORDER_LIGHT,
)


def line_chart(
    data_dict: Dict[str, Tuple[List, List]],
    title: str = "",
    xaxis_title: str = "Time",
    yaxis_title: str = "Value",
    **kwargs
) -> go.Figure:
    """
    Create a multi-line chart.

    Args:
        data_dict: Dictionary where keys are series names and values are
                   tuples of (x_array, y_array).
        title: Chart title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Line chart with template applied.

    Example:
        line_chart({
            "Portfolio": (dates, portfolio_returns),
            "Benchmark": (dates, spy_returns),
        }, title="Cumulative Returns")
    """
    fig = create_figure(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, **kwargs)

    for i, (label, (x, y)) in enumerate(data_dict.items()):
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=label,
            mode="lines",
            line=dict(
                color=CHART_COLORS[i % len(CHART_COLORS)],
                width=2,
            ),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))

    return fig


def area_chart(
    x: List,
    y: List,
    label: str = "Series",
    color: str = ACCENT_BLUE,
    title: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a filled area chart.

    Args:
        x: X-axis values (typically dates).
        y: Y-axis values (typically prices or returns).
        label: Series name for legend.
        color: Hex color for fill.
        title: Chart title.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Area chart.
    """
    fig = create_figure(title=title, **kwargs)

    # Convert hex color to rgba for fill transparency
    hex_color = color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    fill_rgba = f"rgba({r}, {g}, {b}, 0.2)"

    fig.add_trace(go.Scatter(
        x=x, y=y,
        name=label,
        fill="tozeroy",
        line=dict(color=color, width=2),
        fillcolor=fill_rgba,
        hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
    ))

    return fig


def bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    colors: Optional[List[str]] = None,
    horizontal: bool = False,
    **kwargs
) -> go.Figure:
    """
    Create a bar chart (vertical or horizontal).

    Args:
        labels: Category labels.
        values: Bar values.
        title: Chart title.
        colors: Optional list of colors for each bar.
        horizontal: If True, create horizontal bar chart.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Bar chart.
    """
    if colors is None:
        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))]

    fig = create_figure(title=title, **kwargs)

    if horizontal:
        fig.add_trace(go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        ))
    else:
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors),
            hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
        ))

    return fig


def heatmap_chart(
    z: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str = "",
    colorscale: str = "RdBu_r",
    fmt: str = ".2f",
    **kwargs
) -> go.Figure:
    """
    Create an annotated heatmap (e.g., correlation matrix).

    Args:
        z: 2D array of values.
        x_labels: Column labels.
        y_labels: Row labels.
        title: Chart title.
        colorscale: Plotly colorscale name.
        fmt: Format string for annotations.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Heatmap.
    """
    fig = create_figure(title=title, **kwargs)

    # Format annotations
    annotations_text = []
    for row in z:
        row_text = [f"{val:{fmt}}" for val in row]
        annotations_text.append(row_text)

    fig.add_trace(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        text=annotations_text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": TEXT_PRIMARY},
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>%{z:.4f}<extra></extra>",
        colorbar=dict(
            thickness=15,
            tickfont=dict(size=10, color=TEXT_SECONDARY),
        ),
    ))

    return fig


def surface_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "",
    colorscale: str = "Inferno",
    **kwargs
) -> go.Figure:
    """
    Create a 3D surface plot.

    Args:
        X: 2D array of X values (meshgrid).
        Y: 2D array of Y values (meshgrid).
        Z: 2D array of Z values (surface heights).
        title: Chart title.
        colorscale: Plotly colorscale name.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: 3D surface.
    """
    fig = create_figure(title=title, **kwargs)

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        hovertemplate="<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Z:</b> %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor=BORDER_LIGHT),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor=BORDER_LIGHT),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0.1)", gridcolor=BORDER_LIGHT),
        ),
    )

    return fig


def equity_curve(
    dates: List,
    equity: List[float],
    benchmark: Optional[List[float]] = None,
    title: str = "Equity Curve",
    **kwargs
) -> go.Figure:
    """
    Create an equity curve chart with drawdown shading.

    Args:
        dates: List of dates.
        equity: Cumulative equity values.
        benchmark: Optional benchmark series for comparison.
        title: Chart title.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Equity curve with drawdown bands.
    """
    equity_arr = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak

    fig = create_figure(title=title, **kwargs)

    # Drawdown fill
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="rgba(0, 0, 0, 0)"),
        fillcolor=f"rgba(248, 81, 73, 0.15)",
        hovertemplate="<b>Drawdown</b><br>%{x}<br>%{y:.2%}<extra></extra>",
    ))

    # Equity curve
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        name="Portfolio",
        line=dict(color=ACCENT_GREEN, width=2),
        hovertemplate="<b>Portfolio</b><br>%{x}<br>%{y:.0f}<extra></extra>",
    ))

    # Benchmark (optional)
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=benchmark,
            name="Benchmark",
            line=dict(color=ACCENT_BLUE, width=2, dash="dash"),
            hovertemplate="<b>Benchmark</b><br>%{x}<br>%{y:.0f}<extra></extra>",
        ))

    fig.update_yaxes(title_text="Equity / Return")

    return fig


def regime_timeline(
    dates: List,
    regimes: List[int],
    regime_names: Optional[Dict[int, str]] = None,
    title: str = "Regime Timeline",
    **kwargs
) -> go.Figure:
    """
    Create a regime timeline with colored vertical bands.

    Args:
        dates: List of dates.
        regimes: List of regime codes (0, 1, 2, 3, ...).
        regime_names: Optional dict mapping regime code to name.
        title: Chart title.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Timeline with colored regime bands.
    """
    if regime_names is None:
        regime_names = {
            0: "Bull",
            1: "Bear",
            2: "Mean Revert",
            3: "High Vol",
        }

    regime_colors = {
        0: ACCENT_GREEN,
        1: ACCENT_RED,
        2: ACCENT_BLUE,
        3: ACCENT_AMBER,
    }

    fig = create_figure(title=title, **kwargs)

    # Convert to DataFrame for easier grouping
    df = pd.DataFrame({"date": dates, "regime": regimes})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Find regime change points
    regime_changes = df[df["regime"].shift() != df["regime"]].index.tolist()
    regime_changes = [0] + regime_changes + [len(df)]

    for i in range(len(regime_changes) - 1):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i + 1]
        regime = int(df.iloc[start_idx]["regime"])
        color = regime_colors.get(regime, CHART_COLORS[regime % len(CHART_COLORS)])

        start_date = df.iloc[start_idx]["date"]
        end_date = df.iloc[end_idx - 1]["date"]

        fig.add_vrect(
            x0=start_date, x1=end_date,
            fillcolor=color,
            opacity=0.15,
            line_width=0,
            name=regime_names.get(regime, f"Regime {regime}"),
        )

    return fig


def dual_axis_chart(
    x: List,
    y1: List[float],
    y2: List[float],
    label1: str = "Left Axis",
    label2: str = "Right Axis",
    title: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a chart with two y-axes (left and right).

    Args:
        x: X-axis values (typically dates).
        y1: Values for left y-axis.
        y2: Values for right y-axis.
        label1: Name for first series.
        label2: Name for second series.
        title: Chart title.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Dual-axis chart.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=x, y=y1,
            name=label1,
            line=dict(color=ACCENT_BLUE, width=2),
            hovertemplate=f"<b>{label1}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=y2,
            name=label2,
            line=dict(color=ACCENT_GREEN, width=2),
            hovertemplate=f"<b>{label2}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=label1, secondary_y=False)
    fig.update_yaxes(title_text=label2, secondary_y=True)
    fig.update_layout(title_text=title, hovermode="x unified")

    return fig


def candlestick_chart(
    df: pd.DataFrame,
    title: str = "OHLCV Candlestick",
    **kwargs
) -> go.Figure:
    """
    Create a candlestick chart with volume subplot.

    Expects DataFrame with columns: Open, High, Low, Close, Volume.

    Args:
        df: DataFrame with OHLCV data (index is dates).
        title: Chart title.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Candlestick chart with volume subplot.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=ACCENT_GREEN,
            decreasing_line_color=ACCENT_RED,
        ),
        row=1, col=1,
    )

    # Volume
    colors = [ACCENT_GREEN if df.iloc[i]["Close"] >= df.iloc[i]["Open"]
              else ACCENT_RED for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker=dict(color=colors),
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(title_text=title, xaxis_rangeslider_visible=False)

    return fig


def scatter_chart(
    x: List[float],
    y: List[float],
    color: Optional[List[float]] = None,
    size: Optional[List[float]] = None,
    text: Optional[List[str]] = None,
    title: str = "",
    xaxis_title: str = "X",
    yaxis_title: str = "Y",
    **kwargs
) -> go.Figure:
    """
    Create a scatter plot with optional color/size encoding.

    Args:
        x: X-axis values.
        y: Y-axis values.
        color: Optional array for color mapping (creates colorbar).
        size: Optional array for bubble size scaling.
        text: Optional array of hover text.
        title: Chart title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Scatter plot.
    """
    fig = create_figure(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, **kwargs)

    marker_dict = {"color": color if color is not None else ACCENT_BLUE, "size": size if size is not None else 8}
    if color is not None:
        marker_dict["colorscale"] = "Viridis"
        marker_dict["showscale"] = True
        marker_dict["colorbar"] = dict(thickness=15, tickfont=dict(size=10))

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=marker_dict,
        text=text,
        hovertemplate="<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>" if text else
                      "<b>X:</b> %{x:.4f}<br><b>Y:</b> %{y:.4f}<extra></extra>",
    ))

    return fig


def radar_chart(
    categories: List[str],
    values: List[float],
    name: str = "Series",
    title: str = "",
    **kwargs
) -> go.Figure:
    """
    Create a radar/polar chart.

    Args:
        categories: Category names (e.g., metric names).
        values: Metric values (typically normalized to [0, 1]).
        name: Series name for legend.
        title: Chart title.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Radar chart.
    """
    fig = create_figure(title=title, **kwargs)

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name=name,
        line=dict(color=ACCENT_BLUE, width=2),
        fillcolor=f"rgba(88, 166, 255, 0.2)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) if values else 1],
            ),
        ),
    )

    return fig


def histogram_chart(
    values: List[float],
    nbins: int = 30,
    title: str = "",
    color: str = ACCENT_BLUE,
    **kwargs
) -> go.Figure:
    """
    Create a distribution histogram.

    Args:
        values: Array of values to histogram.
        nbins: Number of bins.
        title: Chart title.
        color: Bar color.
        **kwargs: Additional arguments passed to create_figure.

    Returns:
        go.Figure: Histogram.
    """
    fig = create_figure(title=title, **kwargs)

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=nbins,
        name="Distribution",
        marker=dict(color=color),
        hovertemplate="<b>Bin:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Value")
    fig.update_yaxes(title_text="Frequency")

    return fig


__all__ = [
    "line_chart",
    "area_chart",
    "bar_chart",
    "heatmap_chart",
    "surface_3d",
    "equity_curve",
    "regime_timeline",
    "dual_axis_chart",
    "candlestick_chart",
    "scatter_chart",
    "radar_chart",
    "histogram_chart",
]
