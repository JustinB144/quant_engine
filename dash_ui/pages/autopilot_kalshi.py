"""Autopilot & Events -- Strategy lifecycle, paper trading, and Kalshi event markets."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

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
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    empty_figure,
)

try:
    from quant_engine.config import (
        AUTOPILOT_FEATURE_MODE,
        KALSHI_DB_PATH,
        PAPER_STATE_PATH,
        STRATEGY_REGISTRY_PATH,
    )
except ImportError:
    STRATEGY_REGISTRY_PATH = Path("results/autopilot/strategy_registry.json")
    PAPER_STATE_PATH = Path("results/autopilot/paper_state.json")
    AUTOPILOT_FEATURE_MODE = "core"
    KALSHI_DB_PATH = Path("data/kalshi.duckdb")

dash.register_page(__name__, path="/autopilot", name="Autopilot & Events", order=8)

# ---------------------------------------------------------------------------
# Demo data generators (seed=42 for reproducibility)
# ---------------------------------------------------------------------------


def _demo_strategy_candidates():
    """Generate realistic demo strategy candidate data."""
    rng = np.random.RandomState(42)
    names = [
        "MomentumCore_10d_e1.0_c0.6",
        "MeanRevert_5d_e0.8_c0.7",
        "VolBreak_20d_e1.2_c0.5",
        "TrendFollow_10d_e1.0_c0.7_risk",
        "HybridAlpha_10d_e0.8_c0.6",
        "RegimeSwitch_10d_e1.2_c0.6",
        "StatArb_5d_e1.0_c0.5_risk",
        "FactorTilt_20d_e0.8_c0.7",
    ]
    candidates = []
    for name in names:
        sharpe = round(rng.uniform(0.3, 2.2), 2)
        win_rate = round(rng.uniform(0.42, 0.68), 3)
        max_dd = round(rng.uniform(-0.30, -0.05), 3)
        pf = round(rng.uniform(0.8, 2.5), 2)
        dsr_p = round(rng.uniform(0.001, 0.15), 4)
        pbo = round(rng.uniform(0.05, 0.65), 3)
        n_trades = int(rng.randint(40, 500))
        status = rng.choice(["promoted", "passed_sharpe", "passed_dsr", "rejected", "candidate"])
        candidates.append({
            "strategy": name,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "profit_factor": pf,
            "dsr_pvalue": dsr_p,
            "pbo": pbo,
            "n_trades": n_trades,
            "status": status,
        })
    return candidates


def _demo_promotion_funnel():
    """Demo promotion funnel counts."""
    return {
        "Candidates": 24,
        "Passed Sharpe": 18,
        "Passed DSR": 12,
        "Passed PBO": 8,
        "Passed Capacity": 5,
        "Promoted": 3,
    }


def _demo_paper_equity():
    """Generate demo paper trading equity curve."""
    rng = np.random.RandomState(42)
    n = 200
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    returns = rng.normal(0.0008, 0.012, n)
    equity = 1_000_000 * (1 + pd.Series(returns, index=dates)).cumprod()
    return equity


def _demo_paper_positions():
    """Generate demo paper trading positions."""
    rng = np.random.RandomState(42)
    tickers = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM"]
    positions = []
    for t in tickers[:rng.randint(3, 7)]:
        side = rng.choice(["LONG", "SHORT"])
        size = round(rng.uniform(0.02, 0.08), 4)
        entry_px = round(rng.uniform(100, 600), 2)
        current_px = round(entry_px * (1 + rng.normal(0.01, 0.05)), 2)
        pnl = round((current_px / entry_px - 1) * (1 if side == "LONG" else -1) * 100, 2)
        positions.append({
            "ticker": t,
            "side": side,
            "size": f"{size:.2%}",
            "entry_price": entry_px,
            "current_price": current_px,
            "pnl_pct": pnl,
            "strategy": rng.choice(["MomentumCore", "MeanRevert", "VolBreak"]),
        })
    return positions


def _demo_kalshi_events(event_type):
    """Generate demo Kalshi event data for a given type."""
    rng = np.random.RandomState(42 + hash(event_type) % 100)
    n_events = 12

    # Probability distribution
    if event_type == "CPI":
        buckets = ["<2.5%", "2.5-3.0%", "3.0-3.5%", "3.5-4.0%", "4.0-4.5%", ">4.5%"]
        probs = [0.05, 0.15, 0.35, 0.25, 0.12, 0.08]
    elif event_type == "FOMC":
        buckets = ["-50bp", "-25bp", "Hold", "+25bp", "+50bp"]
        probs = [0.02, 0.10, 0.45, 0.35, 0.08]
    elif event_type == "NFP":
        buckets = ["<100K", "100-150K", "150-200K", "200-250K", "250-300K", ">300K"]
        probs = [0.08, 0.15, 0.30, 0.25, 0.15, 0.07]
    else:  # GDP
        buckets = ["<1%", "1-2%", "2-3%", "3-4%", ">4%"]
        probs = [0.10, 0.20, 0.35, 0.25, 0.10]

    # Historical events timeline
    dates = pd.date_range(end=datetime.now(), periods=n_events, freq="MS")
    surprises = rng.normal(0, 0.5, n_events)
    surprise_dirs = ["positive" if s > 0 else "negative" for s in surprises]
    actuals = [round(rng.uniform(0.5, 5.0), 2) for _ in range(n_events)]
    forecasts = [round(a + s * 0.3, 2) for a, s in zip(actuals, surprises)]

    # Walk-forward accuracy
    wf_accuracy = [round(rng.uniform(0.35, 0.85), 3) for _ in range(n_events)]

    # Disagreement signal
    disagreement = rng.normal(0, 1, 60)
    disagreement = pd.Series(
        np.cumsum(disagreement) * 0.05,
        index=pd.bdate_range(end=datetime.now(), periods=60),
    )

    return {
        "buckets": buckets,
        "probs": probs,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "surprise_dirs": surprise_dirs,
        "actuals": actuals,
        "forecasts": forecasts,
        "wf_accuracy": wf_accuracy,
        "disagreement_dates": [d.strftime("%Y-%m-%d") for d in disagreement.index],
        "disagreement_values": disagreement.values.tolist(),
    }


# ---------------------------------------------------------------------------
# Tab 1: Autopilot
# ---------------------------------------------------------------------------

strategy_table_cols = [
    {"name": "Strategy", "id": "strategy"},
    {"name": "Sharpe", "id": "sharpe", "type": "numeric"},
    {"name": "Win Rate", "id": "win_rate", "type": "numeric", "format": dash_table.FormatTemplate.percentage(1)},
    {"name": "Max DD", "id": "max_dd", "type": "numeric", "format": dash_table.FormatTemplate.percentage(1)},
    {"name": "PF", "id": "profit_factor", "type": "numeric"},
    {"name": "DSR p", "id": "dsr_pvalue", "type": "numeric"},
    {"name": "PBO", "id": "pbo", "type": "numeric"},
    {"name": "Trades", "id": "n_trades", "type": "numeric"},
    {"name": "Status", "id": "status"},
]

tab_autopilot = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Run Discovery Cycle", id="ap-discover-btn", className="btn-primary", size="sm"),
                    md="auto",
                ),
                dbc.Col(
                    dbc.Button("View Registry", id="ap-registry-btn", className="btn-secondary", size="sm"),
                    md="auto",
                ),
                dbc.Col(
                    html.Span(
                        id="ap-feature-mode-badge",
                        className="status-badge status-info",
                        style={"marginLeft": "12px"},
                    ),
                    md="auto",
                ),
            ],
            className="g-2 mb-3 align-items-center",
        ),
        dcc.Loading(
            dash_table.DataTable(
                id="ap-strategy-table",
                columns=strategy_table_cols,
                data=[],
                page_size=15,
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
                    {"if": {"filter_query": '{status} = "promoted"'}, "backgroundColor": "rgba(63, 185, 80, 0.08)"},
                    {"if": {"filter_query": '{status} = "rejected"'}, "backgroundColor": "rgba(248, 81, 73, 0.06)"},
                    {
                        "if": {"filter_query": "{sharpe} > 1.0", "column_id": "sharpe"},
                        "color": ACCENT_GREEN,
                    },
                    {
                        "if": {"filter_query": "{sharpe} < 0.75", "column_id": "sharpe"},
                        "color": ACCENT_RED,
                    },
                ],
            )
        ),
        html.Div(style={"height": "16px"}),
        dcc.Loading(dcc.Graph(id="ap-funnel-chart", figure=empty_figure("Run discovery to see funnel"))),
    ],
    style={"paddingTop": "16px"},
)

# ---------------------------------------------------------------------------
# Tab 2: Paper Trading
# ---------------------------------------------------------------------------

paper_position_cols = [
    {"name": "Ticker", "id": "ticker"},
    {"name": "Side", "id": "side"},
    {"name": "Size", "id": "size"},
    {"name": "Entry Px", "id": "entry_price", "type": "numeric"},
    {"name": "Current Px", "id": "current_price", "type": "numeric"},
    {"name": "PnL %", "id": "pnl_pct", "type": "numeric"},
    {"name": "Strategy", "id": "strategy"},
]

tab_paper = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Load Paper State", id="paper-load-btn", className="btn-primary", size="sm"),
                    md="auto",
                ),
            ],
            className="g-2 mb-3",
        ),
        dcc.Loading(dcc.Graph(id="paper-equity-chart", figure=empty_figure("Click Load Paper State"))),
        html.Div(style={"height": "16px"}),
        dcc.Loading(
            dash_table.DataTable(
                id="paper-positions-table",
                columns=paper_position_cols,
                data=[],
                page_size=15,
                sort_action="native",
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
                    {"if": {"filter_query": "{pnl_pct} > 0", "column_id": "pnl_pct"}, "color": ACCENT_GREEN},
                    {"if": {"filter_query": "{pnl_pct} < 0", "column_id": "pnl_pct"}, "color": ACCENT_RED},
                    {"if": {"filter_query": '{side} = "LONG"', "column_id": "side"}, "color": ACCENT_GREEN},
                    {"if": {"filter_query": '{side} = "SHORT"', "column_id": "side"}, "color": ACCENT_RED},
                ],
            )
        ),
    ],
    style={"paddingTop": "16px"},
)

# ---------------------------------------------------------------------------
# Tab 3: Kalshi Events
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    {"label": "CPI", "value": "CPI"},
    {"label": "FOMC", "value": "FOMC"},
    {"label": "NFP", "value": "NFP"},
    {"label": "GDP", "value": "GDP"},
]

tab_kalshi = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="kalshi-event-dd",
                        options=EVENT_TYPES,
                        value="CPI",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                    md=2,
                ),
                dbc.Col(
                    dbc.Button("Load Events", id="kalshi-load-btn", className="btn-primary", size="sm"),
                    md="auto",
                ),
            ],
            className="g-2 mb-3 align-items-end",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="kalshi-prob-chart", figure=empty_figure("Select event type and load"))),
                    md=6,
                ),
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="kalshi-timeline-chart", figure=empty_figure("No event timeline"))),
                    md=6,
                ),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="kalshi-wf-chart", figure=empty_figure("No walk-forward data"))),
                    md=6,
                ),
                dbc.Col(
                    dcc.Loading(dcc.Graph(id="kalshi-disagree-chart", figure=empty_figure("No disagreement signal"))),
                    md=6,
                ),
            ],
            className="g-3",
        ),
    ],
    style={"paddingTop": "16px"},
)

# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        create_page_header("Autopilot & Events", subtitle="Strategy lifecycle and event markets"),
        dcc.Tabs(
            [
                dcc.Tab(tab_autopilot, label="Autopilot", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(tab_paper, label="Paper Trading", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(tab_kalshi, label="Kalshi Events", className="custom-tab", selected_className="custom-tab--selected"),
            ],
            className="custom-tabs",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Autopilot callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("ap-strategy-table", "data"),
    Output("ap-funnel-chart", "figure"),
    Output("ap-feature-mode-badge", "children"),
    Input("ap-discover-btn", "n_clicks"),
    Input("ap-registry-btn", "n_clicks"),
    prevent_initial_call=False,
)
def update_autopilot(discover_clicks, registry_clicks):
    """Run discovery cycle or load registry."""
    feature_mode = "core"
    try:
        feature_mode = AUTOPILOT_FEATURE_MODE
    except (ValueError, KeyError, TypeError, IndexError):
        pass
    badge_text = f"Feature Mode: {feature_mode}"

    # Try loading from real registry first
    candidates = None
    funnel = None
    is_demo = True

    try:
        reg_path = Path(STRATEGY_REGISTRY_PATH)
        if reg_path.exists():
            with open(reg_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            strategies = registry.get("strategies", []) if isinstance(registry, dict) else []
            if strategies:
                candidates = []
                for s in strategies:
                    candidates.append({
                        "strategy": s.get("name", "Unknown"),
                        "sharpe": s.get("sharpe", 0.0),
                        "win_rate": s.get("win_rate", 0.0),
                        "max_dd": s.get("max_drawdown", 0.0),
                        "profit_factor": s.get("profit_factor", 0.0),
                        "dsr_pvalue": s.get("dsr_pvalue", 1.0),
                        "pbo": s.get("pbo", 1.0),
                        "n_trades": s.get("n_trades", 0),
                        "status": s.get("status", "candidate"),
                    })
                # Build funnel from statuses
                status_counts = pd.Series([c["status"] for c in candidates]).value_counts()
                funnel = {
                    "Candidates": len(candidates),
                    "Passed Sharpe": int(status_counts.get("passed_sharpe", 0)) + int(status_counts.get("passed_dsr", 0)) + int(status_counts.get("promoted", 0)),
                    "Passed DSR": int(status_counts.get("passed_dsr", 0)) + int(status_counts.get("promoted", 0)),
                    "Promoted": int(status_counts.get("promoted", 0)),
                }
                is_demo = False
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass

    # Try running autopilot discovery
    if candidates is None and discover_clicks and discover_clicks > 0:
        try:
            from quant_engine.autopilot.discovery import run_discovery_cycle
            result = run_discovery_cycle()
            if result and "candidates" in result:
                candidates = result["candidates"]
                funnel = result.get("funnel", _demo_promotion_funnel())
                is_demo = False
        except (ImportError, ValueError, KeyError, TypeError):
            pass

    # Fall back to demo data
    if candidates is None:
        candidates = _demo_strategy_candidates()
    if funnel is None:
        funnel = _demo_promotion_funnel()

    badge_text += " | DEMO DATA" if is_demo else " | LIVE"

    # Build funnel chart
    funnel_labels = list(funnel.keys())
    funnel_values = list(funnel.values())
    funnel_colors = [ACCENT_BLUE, ACCENT_CYAN, ACCENT_PURPLE, ACCENT_AMBER, ACCENT_GREEN, ACCENT_GREEN][:len(funnel_labels)]

    fig_funnel = go.Figure(go.Bar(
        y=funnel_labels,
        x=funnel_values,
        orientation="h",
        marker_color=funnel_colors,
        text=[str(v) for v in funnel_values],
        textposition="auto",
        textfont=dict(color=TEXT_PRIMARY, size=12, family="Menlo, monospace"),
    ))
    fig_funnel.update_layout(
        title="Strategy Promotion Funnel",
        height=320,
        xaxis=dict(title="Count"),
        yaxis=dict(autorange="reversed"),
    )

    return candidates, fig_funnel, badge_text


# ---------------------------------------------------------------------------
# Paper Trading callback
# ---------------------------------------------------------------------------


@callback(
    Output("paper-equity-chart", "figure"),
    Output("paper-positions-table", "data"),
    Input("paper-load-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_paper_trading(n_clicks):
    """Load paper trading state or use demo data."""
    equity = None
    positions = None

    # Try loading from real paper state
    try:
        state_path = Path(PAPER_STATE_PATH)
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Parse equity curve
            eq_data = state.get("equity_curve", [])
            if eq_data:
                eq_df = pd.DataFrame(eq_data)
                if "date" in eq_df.columns and "equity" in eq_df.columns:
                    eq_df["date"] = pd.to_datetime(eq_df["date"])
                    equity = pd.Series(eq_df["equity"].values, index=eq_df["date"])
            # Parse positions
            pos_data = state.get("positions", [])
            if pos_data:
                positions = pos_data
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass

    # Fallback to demo
    if equity is None:
        equity = _demo_paper_equity()
    if positions is None:
        positions = _demo_paper_positions()

    # Build equity chart
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode="lines",
        name="Paper Equity",
        line=dict(color=ACCENT_BLUE, width=2),
        fill="tonexty" if len(equity) > 1 else None,
    ))
    # Starting capital line
    start_val = equity.iloc[0] if len(equity) > 0 else 1_000_000
    fig_eq.add_hline(
        y=start_val,
        line=dict(color=TEXT_TERTIARY, width=1, dash="dot"),
        annotation_text=f"Start: ${start_val:,.0f}",
        annotation_font_color=TEXT_TERTIARY,
    )
    current_val = equity.iloc[-1] if len(equity) > 0 else start_val
    pnl = current_val - start_val
    pnl_pct = pnl / start_val * 100 if start_val > 0 else 0

    fig_eq.update_layout(
        title=f"Paper Trading Equity -- Current: ${current_val:,.0f} ({pnl_pct:+.1f}%)",
        height=380,
        yaxis=dict(title="Portfolio Value ($)", tickformat="$,.0f"),
    )

    return fig_eq, positions


# ---------------------------------------------------------------------------
# Kalshi Events callback
# ---------------------------------------------------------------------------


@callback(
    Output("kalshi-prob-chart", "figure"),
    Output("kalshi-timeline-chart", "figure"),
    Output("kalshi-wf-chart", "figure"),
    Output("kalshi-disagree-chart", "figure"),
    Input("kalshi-load-btn", "n_clicks"),
    State("kalshi-event-dd", "value"),
    prevent_initial_call=True,
)
def load_kalshi_events(n_clicks, event_type):
    """Load Kalshi event data or use demo data."""
    event_data = None

    # Try loading from real Kalshi DB
    try:
        db_path = Path(KALSHI_DB_PATH)
        if db_path.exists():
            import duckdb

            conn = duckdb.connect(str(db_path), read_only=True)
            # Try to read probability snapshots
            prob_df = conn.execute(
                "SELECT * FROM probability_snapshots WHERE event_type = ? ORDER BY snapshot_ts DESC LIMIT 100",
                [event_type],
            ).fetchdf()
            if not prob_df.empty:
                buckets = prob_df["bucket"].unique().tolist()
                latest = prob_df.groupby("bucket")["yes_price"].last()
                probs = [latest.get(b, 0.0) for b in buckets]

                # Historical events
                try:
                    hist_df = conn.execute(
                        "SELECT * FROM event_outcomes WHERE event_type = ? ORDER BY event_date",
                        [event_type],
                    ).fetchdf()
                    dates = hist_df["event_date"].tolist()
                    surprise_dirs = hist_df.get("surprise_direction", pd.Series(["positive"] * len(dates))).tolist()
                    actuals = hist_df.get("actual", pd.Series([0.0] * len(dates))).tolist()
                    forecasts = hist_df.get("forecast", pd.Series([0.0] * len(dates))).tolist()
                    wf_accuracy = hist_df.get("wf_accuracy", pd.Series([0.5] * len(dates))).tolist()
                except (ValueError, KeyError, TypeError):
                    dates, surprise_dirs, actuals, forecasts, wf_accuracy = [], [], [], [], []

                event_data = {
                    "buckets": buckets,
                    "probs": probs,
                    "dates": dates,
                    "surprise_dirs": surprise_dirs,
                    "actuals": actuals,
                    "forecasts": forecasts,
                    "wf_accuracy": wf_accuracy,
                    "disagreement_dates": [],
                    "disagreement_values": [],
                }
            conn.close()
    except (OSError, ValueError, KeyError, TypeError):
        pass

    # Fallback to demo
    if event_data is None:
        event_data = _demo_kalshi_events(event_type)

    buckets = event_data["buckets"]
    probs = event_data["probs"]
    dates = event_data["dates"]
    surprise_dirs = event_data["surprise_dirs"]
    wf_accuracy = event_data["wf_accuracy"]
    disagree_dates = event_data["disagreement_dates"]
    disagree_vals = event_data["disagreement_values"]

    # --- Probability distribution bar ---
    bar_colors = [ACCENT_BLUE if p < max(probs) else ACCENT_GREEN for p in probs]
    fig_prob = go.Figure(go.Bar(
        x=buckets,
        y=probs,
        marker_color=bar_colors,
        text=[f"{p:.0%}" for p in probs],
        textposition="auto",
        textfont=dict(color=TEXT_PRIMARY, size=11),
    ))
    fig_prob.update_layout(
        title=f"{event_type} -- Current Probability Distribution",
        height=320,
        xaxis=dict(title="Outcome Bucket"),
        yaxis=dict(title="Probability", tickformat=".0%"),
    )

    # --- Event timeline (green/red for surprise direction) ---
    if dates:
        timeline_colors = [ACCENT_GREEN if d == "positive" else ACCENT_RED for d in surprise_dirs]
        actuals = event_data["actuals"]
        forecasts = event_data["forecasts"]
        surprise_vals = [a - f for a, f in zip(actuals, forecasts)]

        fig_timeline = go.Figure(go.Bar(
            x=dates,
            y=surprise_vals,
            marker_color=timeline_colors,
            name="Surprise (Actual - Forecast)",
            text=[f"{v:+.2f}" for v in surprise_vals],
            textposition="auto",
            textfont=dict(size=10),
        ))
        fig_timeline.update_layout(
            title=f"{event_type} -- Surprise History",
            height=320,
            xaxis=dict(title="Event Date"),
            yaxis=dict(title="Surprise (Actual - Forecast)"),
        )
    else:
        fig_timeline = empty_figure("No event history available")

    # --- Walk-forward accuracy bar ---
    if dates and wf_accuracy:
        wf_colors = [ACCENT_GREEN if a >= 0.5 else ACCENT_RED for a in wf_accuracy]
        fig_wf = go.Figure(go.Bar(
            x=dates,
            y=wf_accuracy,
            marker_color=wf_colors,
            text=[f"{a:.0%}" for a in wf_accuracy],
            textposition="auto",
            textfont=dict(size=10),
        ))
        fig_wf.add_hline(
            y=0.5,
            line=dict(color=ACCENT_AMBER, width=1, dash="dash"),
            annotation_text="50% baseline",
            annotation_font_color=ACCENT_AMBER,
        )
        fig_wf.update_layout(
            title=f"{event_type} -- Walk-Forward Accuracy",
            height=320,
            xaxis=dict(title="Event Date"),
            yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1]),
        )
    else:
        fig_wf = empty_figure("No walk-forward data")

    # --- Disagreement signal line chart ---
    if disagree_dates and disagree_vals:
        fig_disagree = go.Figure()
        fig_disagree.add_trace(go.Scatter(
            x=disagree_dates,
            y=disagree_vals,
            mode="lines",
            name="Disagreement Signal",
            line=dict(color=ACCENT_PURPLE, width=2),
        ))
        fig_disagree.add_hline(
            y=0,
            line=dict(color=TEXT_TERTIARY, width=0.5, dash="dot"),
        )
        # Color fill above/below zero
        vals_arr = np.array(disagree_vals)
        above = np.where(vals_arr >= 0, vals_arr, 0).tolist()
        below = np.where(vals_arr < 0, vals_arr, 0).tolist()
        fig_disagree.add_trace(go.Scatter(
            x=disagree_dates,
            y=above,
            fill="tozeroy",
            fillcolor="rgba(63, 185, 80, 0.12)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_disagree.add_trace(go.Scatter(
            x=disagree_dates,
            y=below,
            fill="tozeroy",
            fillcolor="rgba(248, 81, 73, 0.12)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_disagree.update_layout(
            title=f"{event_type} -- Market vs Model Disagreement",
            height=320,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Disagreement Signal"),
        )
    else:
        fig_disagree = empty_figure("No disagreement signal data")

    return fig_prob, fig_timeline, fig_wf, fig_disagree
