"""
Signal Desk -- Prediction generation and signal ranking.

Interactive signal generation with model predictions, confidence scoring,
distribution analysis, and action-oriented signal table with conditional formatting.
"""
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate

from quant_engine.dash_ui.components.page_header import create_page_header
from quant_engine.dash_ui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BORDER,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_CYAN,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    CHART_COLORS, REGIME_COLORS, REGIME_NAMES,
    empty_figure, enhance_time_series,
)

dash.register_page(__name__, path="/signal-desk", name="Signal Desk", order=4)

# ---------------------------------------------------------------------------
# Config imports
# ---------------------------------------------------------------------------
try:
    from quant_engine.config import UNIVERSE_QUICK, ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD
except ImportError:
    UNIVERSE_QUICK = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA",
        "JPM", "UNH", "HD", "V", "DDOG", "CRWD", "CAVA",
    ]
    ENTRY_THRESHOLD = 0.005
    CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_CARD_STYLE = {
    "backgroundColor": BG_SECONDARY,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "16px",
}

_LABEL_STYLE = {
    "fontSize": "11px",
    "color": TEXT_TERTIARY,
    "marginBottom": "4px",
    "fontFamily": "Menlo, monospace",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div([
    # Stores
    dcc.Store(id="sd-signals-data", data=None),

    create_page_header("Signal Desk", subtitle="Prediction generation and signal ranking"),

    # ── Controls row ──────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Horizon", style=_LABEL_STYLE),
            dcc.Dropdown(
                id="sd-horizon-dropdown",
                options=[
                    {"label": "5-day", "value": 5},
                    {"label": "10-day", "value": 10},
                    {"label": "20-day", "value": 20},
                ],
                value=10,
                clearable=False,
                style={"fontSize": "12px"},
            ),
        ], width=2),
        dbc.Col([
            html.Label("Top-N Signals", style=_LABEL_STYLE),
            dbc.Input(
                id="sd-topn-input",
                type="number",
                value=20,
                min=5,
                max=100,
                step=1,
                size="sm",
                style={"backgroundColor": BG_PRIMARY, "borderColor": BORDER,
                       "color": TEXT_PRIMARY, "fontSize": "12px"},
            ),
        ], width=2),
        dbc.Col([
            html.Label("\u00a0", style=_LABEL_STYLE),
            dbc.Button(
                "Generate Signals",
                id="sd-generate-btn",
                color="primary",
                size="sm",
            ),
        ], width=2),
        dbc.Col([
            html.Label("\u00a0", style=_LABEL_STYLE),
            html.Div(id="sd-status-text", style={
                "fontSize": "11px", "color": TEXT_TERTIARY,
                "fontFamily": "Menlo, monospace", "paddingTop": "6px",
            }, children="Ready. Configure parameters and click Generate Signals."),
        ], width=6),
    ], className="mb-3"),

    # ── Signal Table ──────────────────────────────────────────────────
    html.Div([
        html.Div("Signal Rankings", style={
            "fontSize": "13px", "fontWeight": "600",
            "color": TEXT_PRIMARY, "marginBottom": "12px",
            "paddingBottom": "8px", "borderBottom": f"1px solid {BORDER}",
        }),
        dcc.Loading(
            html.Div(
                id="sd-signal-table-container",
                children=html.Div(
                    "No signals generated. Click Generate Signals to start.",
                    style={"fontSize": "11px", "color": TEXT_TERTIARY, "padding": "20px",
                           "textAlign": "center"},
                ),
            ),
            type="dot",
            color=ACCENT_BLUE,
        ),
    ], style=_CARD_STYLE),

    # ── Charts row ────────────────────────────────────────────────────
    dbc.Row([
        # Signal distribution histogram
        dbc.Col([
            html.Div([
                html.Div("Signal Distribution", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": TEXT_PRIMARY, "marginBottom": "8px",
                }),
                dcc.Loading(
                    dcc.Graph(
                        id="sd-distribution-chart",
                        figure=empty_figure("Generate signals to view distribution"),
                        config={"displayModeBar": True},
                        style={"height": "350px"},
                    ),
                    type="dot",
                    color=ACCENT_BLUE,
                ),
            ], style=_CARD_STYLE),
        ], width=6),

        # Confidence vs predicted return scatter
        dbc.Col([
            html.Div([
                html.Div("Confidence vs Predicted Return", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": TEXT_PRIMARY, "marginBottom": "8px",
                }),
                dcc.Loading(
                    dcc.Graph(
                        id="sd-scatter-chart",
                        figure=empty_figure("Generate signals to view scatter plot"),
                        config={"displayModeBar": True},
                        style={"height": "350px"},
                    ),
                    type="dot",
                    color=ACCENT_BLUE,
                ),
            ], style=_CARD_STYLE),
        ], width=6),
    ]),
])


# ---------------------------------------------------------------------------
# Helper: generate demo signals
# ---------------------------------------------------------------------------
def _generate_demo_signals(tickers, horizon, top_n):
    """Generate realistic demo signal data for the given tickers."""
    rng = np.random.RandomState(horizon + 42)
    n = len(tickers)

    predicted_returns = rng.normal(0.0, 0.025, n)
    confidences = rng.beta(5, 3, n)
    confidences = np.clip(confidences, 0.1, 0.99)
    regime_ids = rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.15, 0.3, 0.15])

    # Composite score: combine predicted return magnitude with confidence
    scores = np.abs(predicted_returns) * confidences * 100
    scores = np.round(scores, 2)

    # Determine action
    actions = []
    for pr, conf in zip(predicted_returns, confidences):
        if abs(pr) < ENTRY_THRESHOLD or conf < CONFIDENCE_THRESHOLD:
            actions.append("PASS")
        elif pr > 0:
            actions.append("LONG")
        else:
            actions.append("SHORT")

    regime_labels = [REGIME_NAMES.get(int(r), f"Regime {r}") for r in regime_ids]

    df = pd.DataFrame({
        "Ticker": tickers,
        "Predicted Return": np.round(predicted_returns * 100, 3),
        "Confidence": np.round(confidences, 3),
        "Action": actions,
        "Regime": regime_labels,
        "Score": scores,
        "regime_id": regime_ids,
    })

    # Sort by score descending and take top N
    df = df.sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)
    return df


def _try_live_signals(tickers, horizon, top_n):
    """Attempt to generate real signals using the model pipeline."""
    try:
        from quant_engine.features.pipeline import FeaturePipeline
        from quant_engine.config import DATA_CACHE_DIR, MODEL_DIR
        from quant_engine.models.predictor import ModelPredictor

        predictor = ModelPredictor(model_dir=Path(MODEL_DIR), horizon=int(horizon))
        pipeline = FeaturePipeline(feature_mode="core", verbose=False)

        results = []
        for ticker in tickers:
            cache_path = Path(DATA_CACHE_DIR) / f"{ticker}_1d.parquet"
            if not cache_path.exists():
                continue
            try:
                panel = pd.read_parquet(cache_path)
                panel = panel[["Open", "High", "Low", "Close", "Volume"]].copy()
                panel.index = pd.to_datetime(panel.index)
                panel = panel.sort_index().tail(500)
                features, _ = pipeline.compute(panel, compute_targets_flag=False)
                features = features.replace([np.inf, -np.inf], np.nan).dropna()
                if features.empty:
                    continue
                latest = features.iloc[[-1]]
                pred = predictor.predict(latest)
                if pred is not None and len(pred) > 0:
                    predicted_return = float(pred.get("predicted_return", pred.iloc[0]) if hasattr(pred, "get") else pred[0])
                    confidence = float(pred.get("confidence", 0.5) if hasattr(pred, "get") else 0.5)
                    regime = int(pred.get("regime", 0) if hasattr(pred, "get") else 0)
                    results.append({
                        "Ticker": ticker,
                        "predicted_return_raw": predicted_return,
                        "confidence_raw": confidence,
                        "regime_id": regime,
                    })
            except (ValueError, KeyError, TypeError):
                continue

        if len(results) < 3:
            raise ValueError("Not enough live predictions")

        df = pd.DataFrame(results)
        df["Predicted Return"] = np.round(df["predicted_return_raw"] * 100, 3)
        df["Confidence"] = np.round(df["confidence_raw"], 3)

        actions = []
        for _, row in df.iterrows():
            pr = row["predicted_return_raw"]
            conf = row["confidence_raw"]
            if abs(pr) < ENTRY_THRESHOLD or conf < CONFIDENCE_THRESHOLD:
                actions.append("PASS")
            elif pr > 0:
                actions.append("LONG")
            else:
                actions.append("SHORT")
        df["Action"] = actions
        df["Regime"] = df["regime_id"].map(lambda r: REGIME_NAMES.get(int(r), f"Regime {r}"))
        df["Score"] = np.round(np.abs(df["predicted_return_raw"]) * df["confidence_raw"] * 100, 2)
        df = df.sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)
        df = df[["Ticker", "Predicted Return", "Confidence", "Action", "Regime", "Score", "regime_id"]]
        return df, True

    except (ImportError, OSError, ValueError, KeyError, TypeError):
        return None, False


# ---------------------------------------------------------------------------
# Callback: Generate signals
# ---------------------------------------------------------------------------
@callback(
    Output("sd-signal-table-container", "children"),
    Output("sd-distribution-chart", "figure"),
    Output("sd-scatter-chart", "figure"),
    Output("sd-status-text", "children"),
    Output("sd-signals-data", "data"),
    Input("sd-generate-btn", "n_clicks"),
    State("sd-horizon-dropdown", "value"),
    State("sd-topn-input", "value"),
    prevent_initial_call=True,
)
def generate_signals(n_clicks, horizon, top_n):
    """Generate live/demo signals and populate the table plus diagnostic signal charts."""
    if not n_clicks:
        raise PreventUpdate

    horizon = int(horizon) if horizon else 10
    top_n = int(top_n) if top_n else 20
    top_n = max(5, min(top_n, 100))

    tickers = list(UNIVERSE_QUICK)

    # Try live predictions first, fallback to demo
    live_df, is_live = _try_live_signals(tickers, horizon, top_n)
    if is_live and live_df is not None:
        df = live_df
        source = "live model"
        is_demo = False
    else:
        df = _generate_demo_signals(tickers, horizon, top_n)
        source = "demo"
        is_demo = True

    n_long = int((df["Action"] == "LONG").sum())
    n_short = int((df["Action"] == "SHORT").sum())
    n_pass = int((df["Action"] == "PASS").sum())
    status = (
        f"{source.upper()} | {horizon}d horizon | "
        f"{len(df)} signals: {n_long} LONG, {n_short} SHORT, {n_pass} PASS"
    )

    # Build the DataTable
    table_df = df[["Ticker", "Predicted Return", "Confidence", "Action", "Regime", "Score"]].copy()
    table_df["Predicted Return"] = table_df["Predicted Return"].apply(lambda x: f"{x:+.3f}%")
    table_df["Confidence"] = table_df["Confidence"].apply(lambda x: f"{x:.3f}")
    table_df["Score"] = table_df["Score"].apply(lambda x: f"{x:.2f}")

    table = dash_table.DataTable(
        id="sd-signal-table",
        columns=[
            {"name": "Ticker", "id": "Ticker"},
            {"name": "Predicted Return", "id": "Predicted Return"},
            {"name": "Confidence", "id": "Confidence"},
            {"name": "Action", "id": "Action"},
            {"name": "Regime", "id": "Regime"},
            {"name": "Score", "id": "Score"},
        ],
        data=table_df.to_dict("records"),
        sort_action="native",
        sort_mode="multi",
        page_size=20,
        style_table={
            "overflowX": "auto",
        },
        style_header={
            "backgroundColor": BG_TERTIARY,
            "color": TEXT_PRIMARY,
            "fontWeight": "600",
            "fontSize": "11px",
            "fontFamily": "Menlo, monospace",
            "borderBottom": f"2px solid {BORDER}",
            "textAlign": "left",
            "padding": "10px 12px",
        },
        style_cell={
            "backgroundColor": BG_SECONDARY,
            "color": TEXT_SECONDARY,
            "fontSize": "11px",
            "fontFamily": "Menlo, monospace",
            "border": f"1px solid {BORDER}",
            "textAlign": "left",
            "padding": "8px 12px",
            "minWidth": "80px",
        },
        style_data_conditional=[
            # LONG rows: green accent
            {
                "if": {
                    "filter_query": '{Action} = "LONG"',
                    "column_id": "Action",
                },
                "color": ACCENT_GREEN,
                "fontWeight": "700",
            },
            {
                "if": {
                    "filter_query": '{Action} = "LONG"',
                    "column_id": "Predicted Return",
                },
                "color": ACCENT_GREEN,
            },
            # SHORT rows: red accent
            {
                "if": {
                    "filter_query": '{Action} = "SHORT"',
                    "column_id": "Action",
                },
                "color": ACCENT_RED,
                "fontWeight": "700",
            },
            {
                "if": {
                    "filter_query": '{Action} = "SHORT"',
                    "column_id": "Predicted Return",
                },
                "color": ACCENT_RED,
            },
            # PASS rows: dimmed
            {
                "if": {
                    "filter_query": '{Action} = "PASS"',
                    "column_id": "Action",
                },
                "color": TEXT_TERTIARY,
                "fontStyle": "italic",
            },
            # Hover effect
            {
                "if": {"state": "active"},
                "backgroundColor": BG_TERTIARY,
                "border": f"1px solid {ACCENT_BLUE}",
            },
        ],
    )

    # ── Distribution histogram ────────────────────────────────────────
    pred_returns = df["Predicted Return"].values.astype(float)

    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(
        x=pred_returns,
        nbinsx=30,
        marker_color=ACCENT_BLUE,
        opacity=0.75,
        name="Predicted Returns",
    ))

    # Add entry threshold lines
    threshold_pct = ENTRY_THRESHOLD * 100
    dist_fig.add_vline(
        x=threshold_pct, line_dash="dash", line_color=ACCENT_GREEN,
        annotation_text=f"Long entry ({threshold_pct:.1f}%)",
        annotation_position="top right",
        annotation_font=dict(size=9, color=ACCENT_GREEN),
    )
    dist_fig.add_vline(
        x=-threshold_pct, line_dash="dash", line_color=ACCENT_RED,
        annotation_text=f"Short entry ({-threshold_pct:.1f}%)",
        annotation_position="top left",
        annotation_font=dict(size=9, color=ACCENT_RED),
    )

    # Color regions
    dist_fig.add_vrect(
        x0=threshold_pct, x1=max(pred_returns.max(), threshold_pct + 1),
        fillcolor=ACCENT_GREEN, opacity=0.05, line_width=0,
    )
    dist_fig.add_vrect(
        x0=min(pred_returns.min(), -threshold_pct - 1), x1=-threshold_pct,
        fillcolor=ACCENT_RED, opacity=0.05, line_width=0,
    )

    dist_fig.update_layout(
        title=dict(text=f"Predicted Return Distribution ({horizon}d)", font=dict(size=13)),
        xaxis=dict(title="Predicted Return (%)"),
        yaxis=dict(title="Count"),
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False,
        bargap=0.05,
    )

    # ── Confidence vs return scatter ──────────────────────────────────
    scatter_fig = go.Figure()

    regime_ids = df["regime_id"].values.astype(int)
    unique_regimes = sorted(set(regime_ids))

    for regime_id in unique_regimes:
        mask = regime_ids == regime_id
        regime_name = REGIME_NAMES.get(regime_id, f"Regime {regime_id}")
        color = REGIME_COLORS.get(regime_id, ACCENT_BLUE)
        regime_df = df[mask]

        scatter_fig.add_trace(go.Scatter(
            x=regime_df["Predicted Return"].values.astype(float),
            y=regime_df["Confidence"].values.astype(float),
            mode="markers",
            name=regime_name,
            marker=dict(
                size=10,
                color=color,
                opacity=0.8,
                line=dict(width=1, color=BORDER),
            ),
            text=regime_df["Ticker"].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Predicted: %{x:.3f}%<br>"
                "Confidence: %{y:.3f}<br>"
                f"Regime: {regime_name}"
                "<extra></extra>"
            ),
        ))

    # Add quadrant lines
    scatter_fig.add_hline(
        y=CONFIDENCE_THRESHOLD, line_dash="dot", line_color=TEXT_TERTIARY,
        annotation_text=f"Conf threshold ({CONFIDENCE_THRESHOLD:.0%})",
        annotation_position="bottom right",
        annotation_font=dict(size=9, color=TEXT_TERTIARY),
    )
    scatter_fig.add_vline(
        x=threshold_pct, line_dash="dot", line_color=ACCENT_GREEN,
    )
    scatter_fig.add_vline(
        x=-threshold_pct, line_dash="dot", line_color=ACCENT_RED,
    )

    scatter_fig.update_layout(
        title=dict(text="Confidence vs Predicted Return", font=dict(size=13)),
        xaxis=dict(title="Predicted Return (%)"),
        yaxis=dict(title="Confidence", range=[0, 1.05]),
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10),
        ),
    )

    # Serialize signal data for store
    store_data = df.to_dict("records")

    enhance_time_series(dist_fig)
    enhance_time_series(scatter_fig)

    banner = html.Div(
        "Displaying demo data. Train a model and run predictions for live signals.",
        className="demo-data-banner",
    ) if is_demo else None

    table_container = html.Div([banner, table] if banner else [table])

    return table_container, dist_fig, scatter_fig, status, store_data
