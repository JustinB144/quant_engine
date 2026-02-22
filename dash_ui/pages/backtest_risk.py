"""Backtest & Risk -- Equity curves, risk metrics, and trade analysis."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
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
    REGIME_COLORS,
    REGIME_NAMES,
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

    RESULTS_DIR = Path("results")
    DATA_CACHE_DIR = Path("data/cache")

from quant_engine.dash_ui.data.loaders import (
    build_portfolio_returns,
    compute_risk_metrics,
    load_benchmark_returns,
    load_trades,
)

dash.register_page(__name__, path="/backtest-risk", name="Backtest & Risk", order=5)

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_card_style = {
    "backgroundColor": BG_SECONDARY,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "16px",
}

_label_style = {
    "fontSize": "12px",
    "color": TEXT_TERTIARY,
    "marginBottom": "4px",
}


def _control_group(label, component):
    """Render a labeled control block used in the backtest parameter toolbar."""
    return html.Div(
        [html.Label(label, style=_label_style), component],
        style={"marginBottom": "12px"},
    )


# ---------------------------------------------------------------------------
# Backtest config controls
# ---------------------------------------------------------------------------

controls = html.Div(
    [
        _control_group(
            "Holding Period (days)",
            dbc.Input(
                id="bt-holding-period",
                type="number",
                value=10,
                min=1,
                max=60,
                step=1,
                size="sm",
            ),
        ),
        _control_group(
            "Max Positions",
            dbc.Input(
                id="bt-max-positions",
                type="number",
                value=20,
                min=1,
                max=50,
                step=1,
                size="sm",
            ),
        ),
        _control_group(
            "Entry Threshold",
            dbc.Input(
                id="bt-entry-threshold",
                type="number",
                value=0.005,
                min=0.0,
                max=0.10,
                step=0.001,
                size="sm",
            ),
        ),
        _control_group(
            "Position Size",
            dbc.Input(
                id="bt-position-size",
                type="number",
                value=0.05,
                min=0.01,
                max=0.25,
                step=0.01,
                size="sm",
            ),
        ),
        dbc.Checklist(
            id="bt-risk-mgmt",
            options=[{"label": " Risk Management", "value": "enabled"}],
            value=["enabled"],
            style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginBottom": "12px"},
        ),
        dbc.Button(
            "Run Backtest",
            id="bt-run-btn",
            className="btn-primary",
            size="sm",
            style={"width": "100%"},
        ),
    ],
    className="card-panel",
)

# ---------------------------------------------------------------------------
# Metric card row IDs
# ---------------------------------------------------------------------------

METRIC_IDS = [
    "bt-total-return",
    "bt-ann-return",
    "bt-sharpe",
    "bt-sortino",
    "bt-max-dd",
    "bt-win-rate",
    "bt-profit-factor",
    "bt-total-trades",
]


def _metrics_row():
    """Build the placeholder KPI metric-card row shown above backtest charts."""
    return dbc.Row(
        [
            dbc.Col(html.Div(id=mid), xs=6, md=3, lg=True)
            for mid in METRIC_IDS
        ],
        className="g-3 mb-3",
    )


# ---------------------------------------------------------------------------
# Tab 1 -- Backtest
# ---------------------------------------------------------------------------

tab_backtest = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(controls, md=3, lg=2),
                dbc.Col(
                    [
                        _metrics_row(),
                        dcc.Loading(dcc.Graph(id="bt-equity-curve", figure=empty_figure("Run backtest to see equity curve"))),
                        dcc.Loading(dcc.Graph(id="bt-drawdown-chart", figure=empty_figure("Run backtest to see drawdowns"))),
                    ],
                    md=9,
                    lg=10,
                ),
            ],
            className="g-3",
        ),
    ],
    style={"paddingTop": "16px"},
)

# ---------------------------------------------------------------------------
# Tab 2 -- Risk Analytics
# ---------------------------------------------------------------------------

tab_risk = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="risk-var-waterfall", figure=empty_figure("No risk data"))),
                    md=6,
                ),
                dbc.Col(
                    html.Div(
                        html.Pre(
                            id="risk-metrics-text",
                            children="Run backtest to compute risk metrics.",
                            style={
                                "fontFamily": "Menlo, monospace",
                                "fontSize": "12px",
                                "color": TEXT_SECONDARY,
                                "whiteSpace": "pre-wrap",
                                "margin": "0",
                            },
                        ),
                        className="card-panel",
                    ),
                    md=6,
                ),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="risk-rolling-chart", figure=empty_figure("No rolling risk data"))),
                    md=6,
                ),
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="risk-return-dist", figure=empty_figure("No return distribution"))),
                    md=6,
                ),
            ],
            className="g-3",
        ),
    ],
    style={"paddingTop": "16px"},
)

# ---------------------------------------------------------------------------
# Tab 3 -- Trade Analysis
# ---------------------------------------------------------------------------

trade_table_cols = [
    {"name": "Ticker", "id": "ticker"},
    {"name": "Entry", "id": "entry_date"},
    {"name": "Exit", "id": "exit_date"},
    {"name": "Pred", "id": "predicted_return", "type": "numeric", "format": dash_table.FormatTemplate.percentage(2)},
    {"name": "Actual", "id": "actual_return", "type": "numeric", "format": dash_table.FormatTemplate.percentage(2)},
    {"name": "Net", "id": "net_return", "type": "numeric", "format": dash_table.FormatTemplate.percentage(2)},
    {"name": "Regime", "id": "regime"},
    {"name": "Confidence", "id": "confidence", "type": "numeric", "format": dash_table.FormatTemplate.percentage(1)},
    {"name": "Reason", "id": "reason"},
]

tab_trades = html.Div(
    [
        dcc.Loading(
            dash_table.DataTable(
                id="bt-trade-table",
                columns=trade_table_cols,
                data=[],
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": BG_SECONDARY,
                    "color": TEXT_SECONDARY,
                    "border": f"1px solid {BORDER_LIGHT}",
                    "fontFamily": "Menlo, monospace",
                    "fontSize": "11px",
                    "textAlign": "left",
                    "padding": "6px 10px",
                },
                style_header={
                    "backgroundColor": BG_TERTIARY,
                    "color": TEXT_PRIMARY,
                    "fontWeight": "bold",
                    "border": f"1px solid {BORDER}",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{net_return} > 0", "column_id": "net_return"},
                        "color": ACCENT_GREEN,
                    },
                    {
                        "if": {"filter_query": "{net_return} < 0", "column_id": "net_return"},
                        "color": ACCENT_RED,
                    },
                ],
            )
        ),
        html.Div(style={"height": "16px"}),
        dcc.Loading(dcc.Graph(id="bt-regime-perf", figure=empty_figure("No regime data"))),
    ],
    style={"paddingTop": "16px"},
)

# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        create_page_header(
            "Backtest & Risk",
            subtitle="Equity curves, risk metrics, and trade analysis",
            actions=[
                dbc.Button(
                    [html.I(className="fa-solid fa-download", style={"marginRight": "6px"}),
                     "Export CSV"],
                    id="bt-export-btn",
                    className="btn-export",
                    size="sm",
                ),
            ],
        ),
        dcc.Download(id="bt-download-trades"),
        dcc.Store(id="bt-trades-store"),
        dcc.Store(id="bt-returns-store"),
        dcc.Tabs(
            [
                dcc.Tab(tab_backtest, label="Backtest", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(tab_risk, label="Risk Analytics", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(tab_trades, label="Trade Analysis", className="custom-tab", selected_className="custom-tab--selected"),
            ],
            className="custom-tabs",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("bt-trades-store", "data"),
    Output("bt-returns-store", "data"),
    Output("bt-total-return", "children"),
    Output("bt-ann-return", "children"),
    Output("bt-sharpe", "children"),
    Output("bt-sortino", "children"),
    Output("bt-max-dd", "children"),
    Output("bt-win-rate", "children"),
    Output("bt-profit-factor", "children"),
    Output("bt-total-trades", "children"),
    Output("bt-equity-curve", "figure"),
    Output("bt-drawdown-chart", "figure"),
    Input("bt-run-btn", "n_clicks"),
    State("bt-holding-period", "value"),
    State("bt-max-positions", "value"),
    State("bt-entry-threshold", "value"),
    State("bt-position-size", "value"),
    State("bt-risk-mgmt", "value"),
    prevent_initial_call=False,
)
def run_backtest(n_clicks, holding_period, max_positions, entry_thresh, pos_size, risk_mgmt):
    """Load existing backtest results or run a new backtest."""
    try:
        trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
        trades = load_trades(trades_path)
    except (OSError, ValueError):
        trades = pd.DataFrame()

    if trades.empty:
        empty_cards = [
            metric_card("Total Return", "---", ACCENT_BLUE),
            metric_card("Ann. Return", "---", ACCENT_GREEN),
            metric_card("Sharpe", "---", ACCENT_CYAN),
            metric_card("Sortino", "---", ACCENT_PURPLE),
            metric_card("Max DD", "---", ACCENT_RED),
            metric_card("Win Rate", "---", ACCENT_AMBER),
            metric_card("Profit Factor", "---", ACCENT_ORANGE),
            metric_card("Total Trades", "---", TEXT_SECONDARY),
        ]
        return (
            [],
            [],
            *empty_cards,
            empty_figure("No backtest data found. Place backtest_10d_trades.csv in results/"),
            empty_figure("No drawdown data"),
        )

    # Apply filters if button was clicked
    if n_clicks and n_clicks > 0:
        if entry_thresh and "predicted_return" in trades.columns:
            trades = trades[trades["predicted_return"].abs() >= entry_thresh].copy()
        if pos_size and "position_size" in trades.columns:
            trades["position_size"] = pos_size

    # Build portfolio returns
    port_returns = build_portfolio_returns(trades)

    if port_returns.empty:
        port_returns = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])

    # Compute metrics
    try:
        risk = compute_risk_metrics(port_returns)
    except (ValueError, KeyError, TypeError):
        risk = {k: 0.0 for k in [
            "annual_return", "annual_vol", "sharpe", "sortino", "max_drawdown",
            "var95", "cvar95", "var99", "cvar99"]}

    # Trade-level metrics
    n_trades = len(trades)
    if n_trades > 0 and "net_return" in trades.columns:
        net = trades["net_return"].dropna()
        wins = (net > 0).sum()
        losses = (net <= 0).sum()
        win_rate = wins / max(n_trades, 1)
        gross_profit = net[net > 0].sum()
        gross_loss = abs(net[net <= 0].sum())
        profit_factor = gross_profit / max(gross_loss, 1e-10)
        total_return = float((1 + net).prod() - 1)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        total_return = 0.0

    # Build metric cards
    cards = [
        metric_card(
            "Total Return",
            f"{total_return:+.2%}",
            ACCENT_GREEN if total_return >= 0 else ACCENT_RED,
        ),
        metric_card(
            "Ann. Return",
            f"{risk['annual_return']:+.2%}",
            ACCENT_GREEN if risk["annual_return"] >= 0 else ACCENT_RED,
        ),
        metric_card("Sharpe", f"{risk['sharpe']:.2f}", ACCENT_CYAN),
        metric_card("Sortino", f"{risk['sortino']:.2f}", ACCENT_PURPLE),
        metric_card(
            "Max DD",
            f"{risk['max_drawdown']:.2%}",
            ACCENT_RED,
        ),
        metric_card(
            "Win Rate",
            f"{win_rate:.1%}",
            ACCENT_GREEN if win_rate >= 0.5 else ACCENT_AMBER,
        ),
        metric_card(
            "Profit Factor",
            f"{profit_factor:.2f}",
            ACCENT_GREEN if profit_factor >= 1.0 else ACCENT_RED,
        ),
        metric_card("Total Trades", str(n_trades), TEXT_SECONDARY),
    ]

    # --- Equity curve ---
    eq = (1 + port_returns).cumprod()

    # Benchmark
    try:
        bench_ret = load_benchmark_returns(DATA_CACHE_DIR, port_returns.index)
        if not bench_ret.empty:
            common_idx = eq.index.intersection(bench_ret.index)
            if len(common_idx) > 0:
                bench_ret = bench_ret.reindex(common_idx).fillna(0.0)
                bench_eq = (1 + bench_ret).cumprod()
            else:
                bench_eq = pd.Series(dtype=float)
        else:
            bench_eq = pd.Series(dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        bench_eq = pd.Series(dtype=float)

    # Drawdown series
    peak = eq.cummax()
    dd = (eq / peak) - 1.0

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq.index,
        y=eq.values,
        mode="lines",
        name="Strategy",
        line=dict(color=ACCENT_BLUE, width=2),
    ))
    if not bench_eq.empty:
        fig_eq.add_trace(go.Scatter(
            x=bench_eq.index,
            y=bench_eq.values,
            mode="lines",
            name="Benchmark (SPY)",
            line=dict(color=TEXT_TERTIARY, width=1, dash="dash"),
        ))
    # Drawdown shading
    fig_eq.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        name="Drawdown",
        yaxis="y2",
        line=dict(color=ACCENT_RED, width=0),
        fillcolor="rgba(248, 81, 73, 0.15)",
        showlegend=True,
    ))
    fig_eq.update_layout(
        title="Equity Curve",
        height=380,
        yaxis=dict(title="Cumulative Return"),
        yaxis2=dict(
            title="Drawdown",
            overlaying="y",
            side="right",
            showgrid=False,
            tickformat=".0%",
            range=[dd.min() * 1.5 if len(dd) > 0 else -0.5, 0],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    enhance_time_series(fig_eq)

    # --- Drawdown underwater chart ---
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color=ACCENT_RED, width=1.5),
        fillcolor="rgba(248, 81, 73, 0.25)",
    ))
    fig_dd.update_layout(
        title="Drawdown Underwater",
        height=250,
        yaxis=dict(title="Drawdown", tickformat=".1%"),
    )

    # Serialize for stores
    trades_data = trades.to_dict("records") if not trades.empty else []
    returns_data = port_returns.reset_index().rename(columns={"index": "date", 0: "return"}).to_dict("records") if not port_returns.empty else []

    return (
        trades_data,
        returns_data,
        *cards,
        fig_eq,
        fig_dd,
    )


# ---------------------------------------------------------------------------
# Risk Analytics tab callback
# ---------------------------------------------------------------------------

@callback(
    Output("risk-var-waterfall", "figure"),
    Output("risk-metrics-text", "children"),
    Output("risk-rolling-chart", "figure"),
    Output("risk-return-dist", "figure"),
    Input("bt-returns-store", "data"),
)
def update_risk_analytics(returns_data):
    """Compute and display risk analytics from stored returns."""
    if not returns_data:
        return (
            empty_figure("No risk data -- run backtest first"),
            "No risk metrics available.",
            empty_figure("No rolling risk data"),
            empty_figure("No return distribution"),
        )

    try:
        df = pd.DataFrame(returns_data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.index = pd.DatetimeIndex(df.pop("date"), name="date")
            df = df.sort_index()
        col = "return" if "return" in df.columns else df.columns[0]
        returns = df[col].astype(float).dropna()
    except (ValueError, KeyError, TypeError):
        returns = pd.Series(dtype=float)

    if returns.empty or len(returns) < 5:
        return (
            empty_figure("Insufficient return data"),
            "Insufficient data for risk analysis.",
            empty_figure("Insufficient data"),
            empty_figure("Insufficient data"),
        )

    risk = compute_risk_metrics(returns)

    # --- VaR waterfall ---
    var_labels = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%", "Max DD"]
    var_values = [
        risk["var95"],
        risk["cvar95"],
        risk["var99"],
        risk["cvar99"],
        risk["max_drawdown"],
    ]
    var_colors = [ACCENT_AMBER, ACCENT_ORANGE, ACCENT_RED, "#ff4444", ACCENT_RED]

    fig_var = go.Figure(go.Bar(
        y=var_labels,
        x=var_values,
        orientation="h",
        marker_color=var_colors,
        text=[f"{v:.2%}" for v in var_values],
        textposition="auto",
        textfont=dict(color=TEXT_PRIMARY, size=11),
    ))
    fig_var.update_layout(
        title="Value at Risk Waterfall",
        height=280,
        xaxis=dict(title="Daily Return", tickformat=".1%"),
        yaxis=dict(autorange="reversed"),
    )

    # --- Risk metrics text ---
    text_lines = [
        "RISK METRICS SUMMARY",
        "=" * 40,
        f"  Annualized Return:   {risk['annual_return']:>+8.2%}",
        f"  Annualized Vol:      {risk['annual_vol']:>8.2%}",
        f"  Sharpe Ratio:        {risk['sharpe']:>8.2f}",
        f"  Sortino Ratio:       {risk['sortino']:>8.2f}",
        f"  Max Drawdown:        {risk['max_drawdown']:>8.2%}",
        "",
        "VALUE AT RISK",
        "-" * 40,
        f"  VaR (95%):           {risk['var95']:>+8.4f}",
        f"  CVaR (95%):          {risk['cvar95']:>+8.4f}",
        f"  VaR (99%):           {risk['var99']:>+8.4f}",
        f"  CVaR (99%):          {risk['cvar99']:>+8.4f}",
        "",
        "OBSERVATIONS",
        "-" * 40,
        f"  Trading Days:        {len(returns):>8d}",
        f"  Positive Days:       {(returns > 0).sum():>8d}",
        f"  Negative Days:       {(returns < 0).sum():>8d}",
        f"  Best Day:            {returns.max():>+8.4f}",
        f"  Worst Day:           {returns.min():>+8.4f}",
        f"  Skewness:            {float(returns.skew()):>8.2f}",
        f"  Kurtosis:            {float(returns.kurtosis()):>8.2f}",
    ]
    risk_text = "\n".join(text_lines)

    # --- Rolling risk chart (dual-axis: vol + drawdown) ---
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    eq = (1 + returns).cumprod()
    peak = eq.cummax()
    rolling_dd = (eq / peak) - 1.0

    fig_rolling = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rolling.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            name="Rolling 21D Vol (ann.)",
            line=dict(color=ACCENT_BLUE, width=1.5),
        ),
        secondary_y=False,
    )
    fig_rolling.add_trace(
        go.Scatter(
            x=rolling_dd.index,
            y=rolling_dd.values,
            name="Rolling Drawdown",
            line=dict(color=ACCENT_RED, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248, 81, 73, 0.1)",
        ),
        secondary_y=True,
    )
    fig_rolling.update_layout(
        title="Rolling Risk Metrics",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_rolling.update_yaxes(title_text="Annualized Volatility", tickformat=".0%", secondary_y=False)
    fig_rolling.update_yaxes(title_text="Drawdown", tickformat=".0%", secondary_y=True)
    enhance_time_series(fig_rolling)

    # --- Return distribution histogram ---
    arr = returns.values
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=arr,
        nbinsx=60,
        name="Returns",
        marker_color=ACCENT_BLUE,
        opacity=0.75,
    ))
    # VaR / CVaR vertical lines
    fig_dist.add_vline(x=risk["var95"], line=dict(color=ACCENT_AMBER, width=2, dash="dash"),
                       annotation_text="VaR 95%", annotation_position="top left",
                       annotation_font_color=ACCENT_AMBER)
    fig_dist.add_vline(x=risk["cvar95"], line=dict(color=ACCENT_RED, width=2, dash="dash"),
                       annotation_text="CVaR 95%", annotation_position="top left",
                       annotation_font_color=ACCENT_RED)
    fig_dist.update_layout(
        title="Return Distribution",
        height=350,
        xaxis=dict(title="Daily Return", tickformat=".2%"),
        yaxis=dict(title="Frequency"),
        showlegend=False,
    )

    return fig_var, risk_text, fig_rolling, fig_dist


# ---------------------------------------------------------------------------
# Trade Analysis tab callback
# ---------------------------------------------------------------------------

@callback(
    Output("bt-trade-table", "data"),
    Output("bt-regime-perf", "figure"),
    Input("bt-trades-store", "data"),
)
def update_trade_analysis(trades_data):
    """Populate trade table and regime performance chart."""
    if not trades_data:
        return [], empty_figure("No trade data available")

    try:
        trades = pd.DataFrame(trades_data)
    except (ValueError, KeyError, TypeError):
        return [], empty_figure("Error parsing trade data")

    # Table data
    display_cols = [c["id"] for c in trade_table_cols]
    for col in display_cols:
        if col not in trades.columns:
            trades[col] = ""
    table_data = trades[display_cols].to_dict("records")

    # Regime performance grouped bar
    if "regime" not in trades.columns or "net_return" not in trades.columns:
        return table_data, empty_figure("No regime data in trades")

    trades["regime"] = pd.to_numeric(trades["regime"], errors="coerce")
    trades["net_return"] = pd.to_numeric(trades["net_return"], errors="coerce")
    regime_groups = trades.dropna(subset=["regime", "net_return"]).groupby("regime")

    regimes = []
    avg_returns = []
    win_rates = []
    trade_counts = []
    colors = []

    for regime_id, group in regime_groups:
        r_id = int(regime_id)
        regimes.append(REGIME_NAMES.get(r_id, f"Regime {r_id}"))
        avg_returns.append(float(group["net_return"].mean()))
        win_rates.append(float((group["net_return"] > 0).mean()))
        trade_counts.append(len(group))
        colors.append(REGIME_COLORS.get(r_id, ACCENT_BLUE))

    fig_regime = go.Figure()
    fig_regime.add_trace(go.Bar(
        x=regimes,
        y=avg_returns,
        name="Avg Return",
        marker_color=colors,
        text=[f"{v:.2%}" for v in avg_returns],
        textposition="auto",
        textfont=dict(size=11),
    ))
    fig_regime.add_trace(go.Bar(
        x=regimes,
        y=win_rates,
        name="Win Rate",
        marker_color=[c + "88" for c in colors],
        text=[f"{v:.0%}" for v in win_rates],
        textposition="auto",
        textfont=dict(size=11),
        yaxis="y2",
    ))
    fig_regime.update_layout(
        title="Performance by Regime",
        barmode="group",
        height=380,
        yaxis=dict(title="Avg Return", tickformat=".2%"),
        yaxis2=dict(
            title="Win Rate",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0, 1],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return table_data, fig_regime


# ---------------------------------------------------------------------------
# Export callback
# ---------------------------------------------------------------------------

@callback(
    Output("bt-download-trades", "data"),
    Input("bt-export-btn", "n_clicks"),
    State("bt-trades-store", "data"),
    prevent_initial_call=True,
)
def export_trades_csv(n_clicks, trades_data):
    """Export trade log as CSV download."""
    if not n_clicks or not trades_data:
        return dash.no_update
    df = pd.DataFrame(trades_data)
    return dcc.send_data_frame(df.to_csv, "backtest_trades.csv", index=False)
