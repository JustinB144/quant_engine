"""S&P 500 Comparison -- Benchmark tracking, rolling analytics, and animation."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

from quant_engine.dash_ui.components.metric_card import metric_card
from quant_engine.dash_ui.components.page_header import create_page_header
from quant_engine.dash_ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER,
    BORDER_LIGHT,
    CHART_COLORS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    empty_figure,
    enhance_time_series,
)

try:
    from quant_engine.config import DATA_CACHE_DIR, RESULTS_DIR
except ImportError:
    from pathlib import Path

    DATA_CACHE_DIR = Path("data/cache")
    RESULTS_DIR = Path("results")

from quant_engine.dash_ui.data.loaders import (
    build_portfolio_returns,
    compute_risk_metrics,
    load_benchmark_returns,
    load_trades,
)

dash.register_page(__name__, path="/sp-comparison", name="S&P Comparison", order=7)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

PERIOD_OPTIONS = [
    {"label": "1 Year", "value": "1Y"},
    {"label": "2 Years", "value": "2Y"},
    {"label": "3 Years", "value": "3Y"},
    {"label": "5 Years", "value": "5Y"},
    {"label": "All", "value": "ALL"},
]

PERIOD_DAYS = {"1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260, "ALL": 99999}

controls_row = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id="sp-period-dd",
                options=PERIOD_OPTIONS,
                value="ALL",
                clearable=False,
                style={"backgroundColor": BG_PRIMARY, "color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Button(
                "Load & Compare",
                id="sp-load-btn",
                className="btn-primary",
                size="sm",
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Button(
                "Animate",
                id="sp-animate-btn",
                className="btn-secondary",
                size="sm",
            ),
            md=2,
        ),
    ],
    className="g-2 mb-3 align-items-end",
)

METRIC_IDS = [
    "sp-strat-return",
    "sp-bench-return",
    "sp-alpha",
    "sp-beta",
    "sp-correlation",
    "sp-tracking-error",
    "sp-info-ratio",
]

metrics_row = dbc.Row(
    [dbc.Col(html.Div(id=mid), xs=6, md=3, lg=True) for mid in METRIC_IDS],
    className="g-3 mb-3",
)

layout = html.Div(
    [
        create_page_header("S&P 500 Comparison", subtitle="Benchmark tracking and rolling analytics"),
        dcc.Store(id="sp-data-store"),
        dcc.Store(id="sp-frame-store", data={"frame": 0, "total": 0}),
        dcc.Interval(id="animation-interval", interval=40, disabled=True),
        controls_row,
        metrics_row,
        dcc.Loading(dcc.Graph(id="sp-equity-chart", figure=empty_figure("Click Load & Compare"))),
        dbc.Row(
            [
                dbc.Col(dcc.Loading(dcc.Graph(id="sp-corr-chart", figure=empty_figure("No data"))), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="sp-alpha-beta-chart", figure=empty_figure("No data"))), md=6),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Loading(dcc.Graph(id="sp-relative-chart", figure=empty_figure("No data"))), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="sp-dd-chart", figure=empty_figure("No data"))), md=6),
            ],
            className="g-3",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Load & Compare callback
# ---------------------------------------------------------------------------


@callback(
    Output("sp-data-store", "data"),
    Output("sp-strat-return", "children"),
    Output("sp-bench-return", "children"),
    Output("sp-alpha", "children"),
    Output("sp-beta", "children"),
    Output("sp-correlation", "children"),
    Output("sp-tracking-error", "children"),
    Output("sp-info-ratio", "children"),
    Output("sp-equity-chart", "figure"),
    Output("sp-corr-chart", "figure"),
    Output("sp-alpha-beta-chart", "figure"),
    Output("sp-relative-chart", "figure"),
    Output("sp-dd-chart", "figure"),
    Output("sp-frame-store", "data"),
    Input("sp-load-btn", "n_clicks"),
    State("sp-period-dd", "value"),
    prevent_initial_call=True,
)
def load_and_compare(n_clicks, period):
    """Load portfolio and benchmark returns, compute comparison metrics."""
    empty_cards = [
        metric_card("Strategy Return", "---", ACCENT_BLUE),
        metric_card("S&P Return", "---", ACCENT_GREEN),
        metric_card("Alpha", "---", ACCENT_CYAN),
        metric_card("Beta", "---", ACCENT_PURPLE),
        metric_card("Correlation", "---", ACCENT_AMBER),
        metric_card("Tracking Error", "---", ACCENT_ORANGE),
        metric_card("Info Ratio", "---", ACCENT_RED),
    ]
    empty_store = {"frame": 0, "total": 0}
    empties = [empty_figure("No data")] * 5

    # Load trades and build returns
    try:
        trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
        trades = load_trades(trades_path)
        port_returns = build_portfolio_returns(trades)
    except (OSError, ValueError):
        port_returns = pd.Series(dtype=float)

    if port_returns.empty:
        return (None, *empty_cards, *empties, empty_store)

    # Load benchmark
    try:
        bench_returns = load_benchmark_returns(DATA_CACHE_DIR, port_returns.index)
    except (OSError, ValueError):
        bench_returns = pd.Series(0.0, index=port_returns.index)

    # Align indices
    common = port_returns.index.intersection(bench_returns.index)
    if len(common) < 10:
        return (None, *empty_cards, *empties, empty_store)

    strat = port_returns.reindex(common).fillna(0.0).astype(float)
    bench = bench_returns.reindex(common).fillna(0.0).astype(float)

    # Trim to period
    max_days = PERIOD_DAYS.get(period, 99999)
    if len(strat) > max_days:
        strat = strat.iloc[-max_days:]
        bench = bench.reindex(strat.index).fillna(0.0)

    # --- Compute metrics ---
    strat_total = float((1 + strat).prod() - 1)
    bench_total = float((1 + bench).prod() - 1)
    excess = strat - bench

    # Beta / Alpha via OLS
    bench_arr = bench.values
    strat_arr = strat.values
    if np.std(bench_arr) > 1e-10:
        cov_matrix = np.cov(strat_arr, bench_arr)
        beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
    else:
        beta = 0.0
    ann_strat = float((1 + strat).prod() ** (252 / max(len(strat), 1)) - 1)
    ann_bench = float((1 + bench).prod() ** (252 / max(len(bench), 1)) - 1)
    alpha = ann_strat - beta * ann_bench

    corr = float(strat.corr(bench)) if len(strat) > 5 else 0.0
    tracking_error = float(excess.std() * np.sqrt(252))
    info_ratio = float(excess.mean() * 252 / tracking_error) if tracking_error > 1e-10 else 0.0

    cards = [
        metric_card("Strategy Return", f"{strat_total:+.2%}", ACCENT_GREEN if strat_total >= 0 else ACCENT_RED),
        metric_card("S&P Return", f"{bench_total:+.2%}", ACCENT_GREEN if bench_total >= 0 else ACCENT_RED),
        metric_card("Alpha", f"{alpha:+.2%}", ACCENT_GREEN if alpha > 0 else ACCENT_RED),
        metric_card("Beta", f"{beta:.2f}", ACCENT_CYAN),
        metric_card("Correlation", f"{corr:.2f}", ACCENT_PURPLE),
        metric_card("Tracking Error", f"{tracking_error:.2%}", ACCENT_ORANGE),
        metric_card("Info Ratio", f"{info_ratio:.2f}", ACCENT_GREEN if info_ratio > 0 else ACCENT_RED),
    ]

    # --- Equity curves ---
    strat_eq = (1 + strat).cumprod()
    bench_eq = (1 + bench).cumprod()

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=strat_eq.index, y=strat_eq.values,
        mode="lines", name="Strategy",
        line=dict(color=ACCENT_BLUE, width=2),
    ))
    fig_eq.add_trace(go.Scatter(
        x=bench_eq.index, y=bench_eq.values,
        mode="lines", name="S&P 500 (SPY)",
        line=dict(color=ACCENT_GREEN, width=2),
    ))
    # Fill between
    fig_eq.add_trace(go.Scatter(
        x=list(strat_eq.index) + list(strat_eq.index[::-1]),
        y=list(strat_eq.values) + list(bench_eq.values[::-1]),
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.08)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_eq.update_layout(
        title="Equity Comparison",
        height=380,
        yaxis=dict(title="Cumulative Return"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # --- Rolling 60D correlation ---
    rolling_corr = strat.rolling(60).corr(bench)
    mean_corr = float(rolling_corr.mean())

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(
        x=rolling_corr.index, y=rolling_corr.values,
        mode="lines", name="60D Rolling Corr",
        line=dict(color=ACCENT_PURPLE, width=1.5),
    ))
    fig_corr.add_hline(
        y=mean_corr,
        line=dict(color=ACCENT_AMBER, width=1, dash="dash"),
        annotation_text=f"Mean: {mean_corr:.2f}",
        annotation_font_color=ACCENT_AMBER,
    )
    fig_corr.update_layout(
        title="Rolling 60D Correlation",
        height=320,
        yaxis=dict(title="Correlation", range=[-1, 1]),
    )

    # --- Rolling alpha/beta (dual axis) ---
    window = 60
    rolling_beta = pd.Series(dtype=float, index=strat.index)
    rolling_alpha = pd.Series(dtype=float, index=strat.index)
    for i in range(window, len(strat)):
        s_window = strat.iloc[i - window:i]
        b_window = bench.iloc[i - window:i]
        if b_window.std() > 1e-10:
            cov_w = np.cov(s_window.values, b_window.values)
            b_val = cov_w[0, 1] / cov_w[1, 1]
        else:
            b_val = 0.0
        a_val = s_window.mean() * 252 - b_val * b_window.mean() * 252
        rolling_beta.iloc[i] = b_val
        rolling_alpha.iloc[i] = a_val

    fig_ab = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ab.add_trace(
        go.Scatter(
            x=rolling_alpha.dropna().index,
            y=rolling_alpha.dropna().values,
            name="Rolling Alpha",
            line=dict(color=ACCENT_GREEN, width=1.5),
        ),
        secondary_y=False,
    )
    fig_ab.add_trace(
        go.Scatter(
            x=rolling_beta.dropna().index,
            y=rolling_beta.dropna().values,
            name="Rolling Beta",
            line=dict(color=ACCENT_CYAN, width=1.5),
        ),
        secondary_y=True,
    )
    fig_ab.update_layout(
        title="Rolling 60D Alpha & Beta",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_ab.update_yaxes(title_text="Alpha (ann.)", tickformat=".1%", secondary_y=False)
    fig_ab.update_yaxes(title_text="Beta", secondary_y=True)

    # --- Relative strength ---
    relative_strength = strat_eq / bench_eq

    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=relative_strength.index,
        y=relative_strength.values,
        mode="lines",
        name="Relative Strength",
        line=dict(color=ACCENT_BLUE, width=1.5),
    ))
    # Fill above/below 1.0
    fig_rel.add_trace(go.Scatter(
        x=relative_strength.index,
        y=[1.0] * len(relative_strength),
        mode="lines",
        showlegend=False,
        line=dict(color=TEXT_TERTIARY, width=0.5, dash="dot"),
    ))
    above = relative_strength.copy()
    below = relative_strength.copy()
    above[above < 1.0] = 1.0
    below[below > 1.0] = 1.0

    fig_rel.add_trace(go.Scatter(
        x=list(above.index) + list(above.index[::-1]),
        y=list(above.values) + [1.0] * len(above),
        fill="toself",
        fillcolor="rgba(63, 185, 80, 0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_rel.add_trace(go.Scatter(
        x=list(below.index) + list(below.index[::-1]),
        y=list(below.values) + [1.0] * len(below),
        fill="toself",
        fillcolor="rgba(248, 81, 73, 0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_rel.update_layout(
        title="Relative Strength (Strategy / S&P)",
        height=320,
        yaxis=dict(title="Ratio"),
    )

    # --- Drawdown comparison ---
    strat_peak = strat_eq.cummax()
    strat_dd = (strat_eq / strat_peak) - 1.0
    bench_peak = bench_eq.cummax()
    bench_dd = (bench_eq / bench_peak) - 1.0

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=strat_dd.index, y=strat_dd.values,
        fill="tozeroy",
        name="Strategy DD",
        line=dict(color=ACCENT_BLUE, width=1),
        fillcolor="rgba(88, 166, 255, 0.2)",
    ))
    fig_dd.add_trace(go.Scatter(
        x=bench_dd.index, y=bench_dd.values,
        fill="tozeroy",
        name="S&P DD",
        line=dict(color=ACCENT_RED, width=1),
        fillcolor="rgba(248, 81, 73, 0.15)",
    ))
    fig_dd.update_layout(
        title="Drawdown Comparison",
        height=320,
        yaxis=dict(title="Drawdown", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Serialize data for animation store
    store_data = {
        "strat_dates": [d.isoformat() for d in strat_eq.index],
        "strat_eq": strat_eq.values.tolist(),
        "bench_eq": bench_eq.values.tolist(),
    }

    frame_store = {"frame": 0, "total": len(strat_eq)}

    enhance_time_series(fig_eq)
    enhance_time_series(fig_corr)
    enhance_time_series(fig_rel)

    return (store_data, *cards, fig_eq, fig_corr, fig_ab, fig_rel, fig_dd, frame_store)


# ---------------------------------------------------------------------------
# Animation toggle
# ---------------------------------------------------------------------------

@callback(
    Output("animation-interval", "disabled"),
    Output("sp-animate-btn", "children"),
    Input("sp-animate-btn", "n_clicks"),
    State("animation-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_animation(n_clicks, currently_disabled):
    """Toggle the animation interval on/off."""
    if currently_disabled:
        return False, "Pause"
    return True, "Animate"


# ---------------------------------------------------------------------------
# Animation frame callback
# ---------------------------------------------------------------------------

@callback(
    Output("sp-equity-chart", "figure", allow_duplicate=True),
    Output("sp-frame-store", "data", allow_duplicate=True),
    Input("animation-interval", "n_intervals"),
    State("sp-data-store", "data"),
    State("sp-frame-store", "data"),
    prevent_initial_call=True,
)
def animate_equity_chart(n_intervals, store_data, frame_data):
    """Progressively reveal data points on the main equity chart."""
    if not store_data or not frame_data:
        return dash.no_update, dash.no_update

    frame = frame_data.get("frame", 0)
    total = frame_data.get("total", 0)

    if total == 0 or frame >= total:
        # Animation complete -- disable
        return dash.no_update, {"frame": total, "total": total}

    # Advance frame
    step = max(1, total // 200)
    next_frame = min(frame + step, total)

    dates = store_data["strat_dates"][:next_frame]
    strat_vals = store_data["strat_eq"][:next_frame]
    bench_vals = store_data["bench_eq"][:next_frame]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=strat_vals,
        mode="lines", name="Strategy",
        line=dict(color=ACCENT_BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=bench_vals,
        mode="lines", name="S&P 500 (SPY)",
        line=dict(color=ACCENT_GREEN, width=2),
    ))
    # Fill between
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=strat_vals + bench_vals[::-1],
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.08)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"Equity Comparison (frame {next_frame}/{total})",
        height=380,
        yaxis=dict(title="Cumulative Return"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision="animation",
    )

    return fig, {"frame": next_frame, "total": total}
