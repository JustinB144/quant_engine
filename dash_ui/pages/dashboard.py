"""
Dashboard -- Portfolio Intelligence Overview.

Main landing page for the Quant Engine Dash UI.  Provides KPI metric cards,
an equity curve, regime state, model health, feature importance, trade log,
and risk analytics in a tabbed layout with auto-refresh.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from plotly.subplots import make_subplots

from quant_engine.dash_ui.components.metric_card import metric_card
from quant_engine.dash_ui.components.page_header import create_page_header
from quant_engine.dash_ui.data.loaders import (
    build_portfolio_returns,
    compute_health_scores,
    compute_model_health,
    compute_regime_payload,
    compute_risk_metrics,
    load_benchmark_returns,
    load_factor_proxies,
    load_feature_importance,
    load_trades,
)
from quant_engine.dash_ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_PRIMARY,
    BG_SECONDARY,
    BORDER,
    REGIME_COLORS,
    REGIME_NAMES,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    empty_figure,
    enhance_time_series,
)

# ---------------------------------------------------------------------------
# Page registration
# ---------------------------------------------------------------------------
dash.register_page(__name__, path="/", name="Dashboard", order=0)

# ---------------------------------------------------------------------------
# Config imports (paths for data loading)
# ---------------------------------------------------------------------------
try:
    from quant_engine.config import DATA_CACHE_DIR, MODEL_DIR, RESULTS_DIR
except ImportError:
    DATA_CACHE_DIR = Path("data/cache")
    MODEL_DIR = Path("trained_models")
    RESULTS_DIR = Path("results")


# ── Helpers ───────────────────────────────────────────────────────────────

def _card_panel(title: str, children, extra_style: dict | None = None):
    """Wrap children inside a styled card-panel div."""
    style = {}
    if extra_style:
        style.update(extra_style)
    return html.Div(
        [
            html.Div(title, className="card-panel-header"),
            html.Div(children),
        ],
        className="card-panel",
        style=style,
    )


def _pct(val: float, decimals: int = 2) -> str:
    """Format a numeric value for display."""
    return f"{val * 100:+.{decimals}f}%"


def _fmt(val: float, decimals: int = 2) -> str:
    """Format a numeric value for display."""
    return f"{val:.{decimals}f}"


# ── Layout ────────────────────────────────────────────────────────────────

layout = html.Div(
    [
        # Hidden stores and interval timer
        dcc.Store(id="dashboard-data"),
        dcc.Interval(id="dashboard-interval", interval=30_000, n_intervals=0),

        # Page header
        create_page_header(
            "Portfolio Intelligence Dashboard",
            actions=[
                html.Span(
                    id="dashboard-last-updated",
                    style={"fontSize": "10px", "color": TEXT_TERTIARY,
                           "fontFamily": "Menlo, monospace"},
                ),
                html.Button(
                    "Refresh",
                    id="dashboard-refresh-btn",
                    className="btn-secondary",
                    style={"padding": "6px 16px", "borderRadius": "6px", "fontSize": "12px"},
                ),
            ],
        ),

        # ── Metric card rows ─────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(html.Div(id="card-portfolio-value"), md=3),
                dbc.Col(html.Div(id="card-30d-return"), md=3),
                dbc.Col(html.Div(id="card-sharpe"), md=3),
                dbc.Col(html.Div(id="card-regime"), md=3),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(id="card-retrain"), md=3),
                dbc.Col(html.Div(id="card-cv-gap"), md=3),
                dbc.Col(html.Div(id="card-data-quality"), md=3),
                dbc.Col(html.Div(id="card-system-health"), md=3),
            ],
            className="g-3 mb-4",
        ),

        # ── Tabs ──────────────────────────────────────────────────────────
        dcc.Tabs(
            id="dashboard-tabs",
            value="tab-portfolio",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Portfolio Overview", value="tab-portfolio",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Regime State", value="tab-regime",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Model Performance", value="tab-model",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Feature Importance", value="tab-features",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Trade Log", value="tab-trades",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Risk Metrics", value="tab-risk",
                        className="custom-tab", selected_className="custom-tab--selected"),
            ],
        ),
        html.Div(id="dashboard-tab-content", style={"marginTop": "20px"}),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────

@callback(
    Output("dashboard-data", "data"),
    Input("dashboard-interval", "n_intervals"),
    Input("dashboard-refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def load_dashboard_data(n_intervals, n_clicks):
    """Load all data and cache in dcc.Store as JSON-safe dict."""
    payload: Dict[str, Any] = {
        "loaded": False,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        cache_dir = Path(DATA_CACHE_DIR)
        model_dir = Path(MODEL_DIR)
        results_dir = Path(RESULTS_DIR)

        # ── Trades ────────────────────────────────────────────────────
        trades_path = results_dir / "backtest_trades.csv"
        trades = load_trades(trades_path)
        portfolio_returns = build_portfolio_returns(trades)

        # ── Risk metrics ──────────────────────────────────────────────
        risk = compute_risk_metrics(portfolio_returns)

        # ── Equity curve ──────────────────────────────────────────────
        if not portfolio_returns.empty:
            equity = (1 + portfolio_returns).cumprod()
            equity_dates = [d.isoformat() for d in equity.index]
            equity_values = equity.tolist()
        else:
            equity_dates = []
            equity_values = []

        # ── Benchmark ─────────────────────────────────────────────────
        benchmark_returns = load_benchmark_returns(cache_dir, portfolio_returns.index)
        if not benchmark_returns.empty:
            bench_eq = (1 + benchmark_returns).cumprod()
            bench_dates = [d.isoformat() for d in bench_eq.index]
            bench_values = bench_eq.tolist()
        else:
            bench_dates = []
            bench_values = []

        # ── Factor attribution ────────────────────────────────────────
        factors = load_factor_proxies(cache_dir, portfolio_returns.index)
        if not portfolio_returns.empty and not factors.empty:
            common = portfolio_returns.index.intersection(factors.index)
            if len(common) > 20:
                y = portfolio_returns.loc[common].values
                X = factors.loc[common].values
                try:
                    from numpy.linalg import lstsq
                    coefs, _, _, _ = lstsq(X, y, rcond=None)
                    residual = y - X @ coefs
                    attribution = {
                        "tech_minus_def": float(coefs[0]) if len(coefs) > 0 else 0.0,
                        "mom_spread": float(coefs[1]) if len(coefs) > 1 else 0.0,
                        "alpha": float(residual.mean()) * 252,
                    }
                except (ValueError, KeyError, TypeError):
                    attribution = {"tech_minus_def": 0.0, "mom_spread": 0.0, "alpha": 0.0}
            else:
                attribution = {"tech_minus_def": 0.0, "mom_spread": 0.0, "alpha": 0.0}
        else:
            attribution = {"tech_minus_def": 0.0, "mom_spread": 0.0, "alpha": 0.0}

        # Attribution text
        attr_text = (
            f"Factor Attribution (OLS decomposition)\n"
            f"---------------------------------------\n"
            f"Tech minus Defensive beta: {attribution['tech_minus_def']:+.4f}\n"
            f"Momentum spread beta:      {attribution['mom_spread']:+.4f}\n"
            f"Annualized alpha:          {attribution['alpha']:+.4f}\n"
        )

        # ── Regime ────────────────────────────────────────────────────
        regime_payload = compute_regime_payload(cache_dir)
        regime_label = regime_payload.get("current_label", "Unavailable")
        regime_as_of = regime_payload.get("as_of", "---")
        current_probs = regime_payload.get("current_probs", {})

        # Serialize probability history
        prob_history = regime_payload.get("prob_history", pd.DataFrame())
        if isinstance(prob_history, pd.DataFrame) and not prob_history.empty:
            prob_hist_dates = [d.isoformat() for d in prob_history.index]
            prob_hist_cols = {}
            for col in prob_history.columns:
                prob_hist_cols[col] = prob_history[col].tolist()
        else:
            prob_hist_dates = []
            prob_hist_cols = {}

        # Serialize transition matrix
        trans = regime_payload.get("transition", np.eye(4))
        trans_list = np.asarray(trans).tolist()

        # ── Model Health ──────────────────────────────────────────────
        model_health = compute_model_health(model_dir, trades)
        cv_gap = model_health.get("cv_gap", 0.0)
        holdout_r2 = model_health.get("holdout_r2", 0.0)
        holdout_ic = model_health.get("holdout_ic", 0.0)
        ic_drift = model_health.get("ic_drift", 0.0)
        retrain_triggered = model_health.get("retrain_triggered", False)
        retrain_reasons = model_health.get("retrain_reasons", [])

        # Registry history
        reg_hist = model_health.get("registry_history", pd.DataFrame())
        if isinstance(reg_hist, pd.DataFrame) and not reg_hist.empty:
            reg_hist_json = reg_hist.to_dict(orient="list")
        else:
            reg_hist_json = {}

        # ── Feature Importance ────────────────────────────────────────
        global_imp, regime_heat = load_feature_importance(model_dir)
        if not global_imp.empty:
            imp_names = global_imp.sort_values(ascending=False).head(20).index.tolist()
            imp_values = global_imp.sort_values(ascending=False).head(20).tolist()
        else:
            imp_names = []
            imp_values = []

        if isinstance(regime_heat, pd.DataFrame) and not regime_heat.empty:
            rh_features = regime_heat.index.tolist()
            rh_regimes = regime_heat.columns.tolist()
            rh_values = regime_heat.values.tolist()
        else:
            rh_features = []
            rh_regimes = []
            rh_values = []

        # ── Trade log ─────────────────────────────────────────────────
        if not trades.empty:
            trade_cols = ["ticker", "entry_date", "exit_date", "predicted_return",
                          "actual_return", "net_return", "regime", "confidence", "reason"]
            existing = [c for c in trade_cols if c in trades.columns]
            trade_records = trades[existing].tail(200).to_dict(orient="records")
        else:
            trade_records = []

        # ── Portfolio value / 30D return ──────────────────────────────
        if equity_values:
            portfolio_value = equity_values[-1]
        else:
            portfolio_value = 1.0

        if not portfolio_returns.empty and len(portfolio_returns) >= 20:
            ret_30d = float((1 + portfolio_returns.tail(20)).prod() - 1)
        elif not portfolio_returns.empty:
            ret_30d = float((1 + portfolio_returns).prod() - 1)
        else:
            ret_30d = 0.0

        # ── Health scores ─────────────────────────────────────────────
        data_quality, system_health = compute_health_scores(model_health, trades, cv_gap)

        # ── Daily returns for risk histograms ─────────────────────────
        if not portfolio_returns.empty:
            daily_returns_list = portfolio_returns.tolist()
            daily_returns_dates = [d.isoformat() for d in portfolio_returns.index]
        else:
            daily_returns_list = []
            daily_returns_dates = []

        # ── Assemble payload ──────────────────────────────────────────
        payload.update({
            "loaded": True,
            "portfolio_value": portfolio_value,
            "ret_30d": ret_30d,
            "sharpe": risk.get("sharpe", 0.0),
            "sortino": risk.get("sortino", 0.0),
            "annual_return": risk.get("annual_return", 0.0),
            "annual_vol": risk.get("annual_vol", 0.0),
            "max_drawdown": risk.get("max_drawdown", 0.0),
            "var95": risk.get("var95", 0.0),
            "cvar95": risk.get("cvar95", 0.0),
            "var99": risk.get("var99", 0.0),
            "cvar99": risk.get("cvar99", 0.0),
            "equity_dates": equity_dates,
            "equity_values": equity_values,
            "bench_dates": bench_dates,
            "bench_values": bench_values,
            "attribution": attribution,
            "attr_text": attr_text,
            "regime_label": regime_label,
            "regime_as_of": regime_as_of,
            "current_probs": current_probs,
            "prob_hist_dates": prob_hist_dates,
            "prob_hist_cols": prob_hist_cols,
            "trans_matrix": trans_list,
            "cv_gap": cv_gap,
            "holdout_r2": holdout_r2,
            "holdout_ic": holdout_ic,
            "ic_drift": ic_drift,
            "retrain_triggered": retrain_triggered,
            "retrain_reasons": retrain_reasons,
            "reg_hist_json": reg_hist_json,
            "imp_names": imp_names,
            "imp_values": imp_values,
            "rh_features": rh_features,
            "rh_regimes": rh_regimes,
            "rh_values": rh_values,
            "trade_records": trade_records,
            "data_quality": data_quality,
            "system_health": system_health,
            "daily_returns_list": daily_returns_list,
            "daily_returns_dates": daily_returns_dates,
            "n_trades": len(trades),
        })

    except (ValueError, KeyError, TypeError, IndexError) as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()

    return payload


# ── Metric card updates ───────────────────────────────────────────────────

@callback(
    Output("card-portfolio-value", "children"),
    Output("card-30d-return", "children"),
    Output("card-sharpe", "children"),
    Output("card-regime", "children"),
    Output("card-retrain", "children"),
    Output("card-cv-gap", "children"),
    Output("card-data-quality", "children"),
    Output("card-system-health", "children"),
    Output("dashboard-last-updated", "children"),
    Input("dashboard-data", "data"),
)
def update_metric_cards(data):
    """Populate the top KPI cards (and header timestamp) from the cached dashboard payload."""
    if not data or not data.get("loaded"):
        blank = metric_card("---", "---")
        return [blank] * 8 + [""]

    pv = data.get("portfolio_value", 1.0)
    pv_str = f"${pv:,.2f}" if pv != 1.0 else "$1.00"
    card_pv = metric_card("Portfolio Value", pv_str, color=ACCENT_BLUE,
                          subtitle="Normalized equity curve")

    r30 = data.get("ret_30d", 0.0)
    r30_color = ACCENT_GREEN if r30 >= 0 else ACCENT_RED
    card_ret = metric_card("30D Return", _pct(r30), color=r30_color,
                           subtitle="Rolling 20-day return")

    sharpe = data.get("sharpe", 0.0)
    s_color = ACCENT_GREEN if sharpe >= 1.0 else ACCENT_AMBER if sharpe >= 0.5 else ACCENT_RED
    card_sharpe = metric_card("Sharpe Ratio", _fmt(sharpe), color=s_color,
                              subtitle=f"Sortino: {_fmt(data.get('sortino', 0.0))}")

    regime = data.get("regime_label", "Unavailable")
    regime_idx = {v: k for k, v in REGIME_NAMES.items()}.get(regime, None)
    regime_color = REGIME_COLORS.get(regime_idx, ACCENT_BLUE) if regime_idx is not None else TEXT_TERTIARY
    # Compute max regime confidence for subtitle
    current_probs = data.get("current_probs", {})
    max_conf = max(current_probs.values()) if current_probs else 0.0
    regime_subtitle = f"{max_conf:.0%} confidence  |  {data.get('regime_as_of', '---')}"
    card_regime = metric_card("Current Regime", regime, color=regime_color,
                              subtitle=regime_subtitle)

    retrain = data.get("retrain_triggered", False)
    rt_val = "TRIGGERED" if retrain else "Stable"
    rt_color = ACCENT_RED if retrain else ACCENT_GREEN
    reasons = data.get("retrain_reasons", [])
    rt_sub = "; ".join(reasons[:2]) if reasons else "No retrain signals"
    card_retrain = metric_card("Retrain Trigger", rt_val, color=rt_color, subtitle=rt_sub)

    cvg = data.get("cv_gap", 0.0)
    cvg_color = ACCENT_GREEN if cvg < 0.05 else ACCENT_AMBER if cvg < 0.15 else ACCENT_RED
    card_cvg = metric_card("CV Gap", _fmt(cvg, 4), color=cvg_color,
                           subtitle="IS vs OOS degradation")

    dq = data.get("data_quality", "---")
    dq_color = (ACCENT_GREEN if dq == "GOOD" else ACCENT_AMBER if dq == "FAIR" else ACCENT_RED)
    card_dq = metric_card("Data Quality", dq, color=dq_color, subtitle="Cache & survivorship")

    sh = data.get("system_health", "---")
    sh_color = (ACCENT_GREEN if sh == "HEALTHY" else ACCENT_AMBER if sh == "CAUTION" else ACCENT_RED)
    card_sh = metric_card("System Health", sh, color=sh_color, subtitle="Composite assessment")

    # Last updated timestamp
    ts = data.get("timestamp", "")
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            last_updated = f"Updated {dt.strftime('%H:%M:%S')}"
        except ValueError:
            last_updated = ""
    else:
        last_updated = ""

    return card_pv, card_ret, card_sharpe, card_regime, card_retrain, card_cvg, card_dq, card_sh, last_updated


# ── Tab content dispatch ──────────────────────────────────────────────────

@callback(
    Output("dashboard-tab-content", "children"),
    Input("dashboard-tabs", "value"),
    Input("dashboard-data", "data"),
)
def render_tab_content(tab, data):
    """Dispatch the active dashboard tab to the matching tab-render helper."""
    if not data or not data.get("loaded"):
        return html.Div(
            "Loading data...",
            style={"color": TEXT_TERTIARY, "padding": "40px", "textAlign": "center"},
        )

    if tab == "tab-portfolio":
        return _render_portfolio_tab(data)
    elif tab == "tab-regime":
        return _render_regime_tab(data)
    elif tab == "tab-model":
        return _render_model_tab(data)
    elif tab == "tab-features":
        return _render_features_tab(data)
    elif tab == "tab-trades":
        return _render_trades_tab(data)
    elif tab == "tab-risk":
        return _render_risk_tab(data)
    return html.Div("Unknown tab")


# ---------------------------------------------------------------------------
# Tab 1: Portfolio Overview
# ---------------------------------------------------------------------------

def _render_portfolio_tab(data: dict):
    # ── Equity curve ──────────────────────────────────────────────────
    """Render the portfolio overview tab (equity curve, attribution chart, and narrative)."""
    eq_dates = data.get("equity_dates", [])
    eq_vals = data.get("equity_values", [])
    b_dates = data.get("bench_dates", [])
    b_vals = data.get("bench_values", [])

    if eq_dates and eq_vals:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_dates, y=eq_vals,
            mode="lines", name="Portfolio",
            line=dict(color=ACCENT_BLUE, width=2),
        ))
        if b_dates and b_vals:
            fig_eq.add_trace(go.Scatter(
                x=b_dates, y=b_vals,
                mode="lines", name="Benchmark (SPY)",
                line=dict(color=TEXT_TERTIARY, width=1, dash="dot"),
            ))
        fig_eq.update_layout(
            title="Cumulative Equity Curve",
            xaxis_title="Date", yaxis_title="Growth of $1",
            height=420, legend=dict(x=0.01, y=0.99),
        )
        enhance_time_series(fig_eq)
    else:
        fig_eq = empty_figure("No equity data available")

    # ── Attribution bar chart ─────────────────────────────────────────
    attr = data.get("attribution", {})
    attr_labels = list(attr.keys())
    attr_values = [attr[k] for k in attr_labels]
    attr_colors = [ACCENT_BLUE if v >= 0 else ACCENT_RED for v in attr_values]

    if any(v != 0 for v in attr_values):
        fig_attr = go.Figure(go.Bar(
            x=attr_labels, y=attr_values,
            marker_color=attr_colors,
            text=[f"{v:+.4f}" for v in attr_values],
            textposition="outside",
        ))
        fig_attr.update_layout(
            title="Return Attribution",
            yaxis_title="Coefficient",
            height=340,
        )
    else:
        fig_attr = empty_figure("No attribution data")

    # ── Attribution text ──────────────────────────────────────────────
    attr_text = data.get("attr_text", "No attribution analysis available.")

    return html.Div([
        _card_panel("Equity Curve", dcc.Graph(figure=fig_eq, config={"displayModeBar": False})),
        dbc.Row([
            dbc.Col(
                _card_panel("Attribution Chart",
                            dcc.Graph(figure=fig_attr, config={"displayModeBar": False})),
                md=7,
            ),
            dbc.Col(
                _card_panel("Attribution Analysis", html.Pre(
                    attr_text,
                    style={"color": TEXT_SECONDARY, "fontSize": "12px",
                           "fontFamily": "Menlo, monospace", "margin": "0",
                           "whiteSpace": "pre-wrap"},
                )),
                md=5,
            ),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 2: Regime State
# ---------------------------------------------------------------------------

def _render_regime_tab(data: dict):
    """Render the regime-state tab with history, current probabilities, and transitions."""
    prob_dates = data.get("prob_hist_dates", [])
    prob_cols = data.get("prob_hist_cols", {})
    current_probs = data.get("current_probs", {})
    trans_matrix = data.get("trans_matrix", [])

    # ── Stacked area probability history ──────────────────────────────
    if prob_dates and prob_cols:
        fig_area = go.Figure()
        for i in range(4):
            col_key = f"regime_prob_{i}"
            if col_key in prob_cols:
                fig_area.add_trace(go.Scatter(
                    x=prob_dates,
                    y=prob_cols[col_key],
                    mode="lines",
                    name=REGIME_NAMES.get(i, f"Regime {i}"),
                    stackgroup="one",
                    line=dict(width=0.5, color=REGIME_COLORS.get(i, ACCENT_BLUE)),
                    fillcolor=REGIME_COLORS.get(i, ACCENT_BLUE),
                ))
        fig_area.update_layout(
            title="Regime Probability History",
            yaxis_title="Probability", yaxis_range=[0, 1],
            height=380, legend=dict(orientation="h", y=-0.15),
        )
        enhance_time_series(fig_area)
    else:
        fig_area = empty_figure("No regime history available")

    # ── Current probabilities horizontal bar ──────────────────────────
    if current_probs:
        labels = list(current_probs.keys())
        values = list(current_probs.values())
        colors = []
        for i, label in enumerate(labels):
            idx = {v: k for k, v in REGIME_NAMES.items()}.get(label, i)
            colors.append(REGIME_COLORS.get(idx, ACCENT_BLUE))
        fig_bar = go.Figure(go.Bar(
            y=labels, x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
        ))
        fig_bar.update_layout(
            title="Current Regime Probabilities",
            xaxis_title="Probability", xaxis_range=[0, 1],
            height=280,
        )
    else:
        fig_bar = empty_figure("No current probabilities")

    # ── Transition matrix heatmap ─────────────────────────────────────
    if trans_matrix and len(trans_matrix) >= 2:
        regime_labels = [REGIME_NAMES.get(i, f"R{i}") for i in range(len(trans_matrix))]
        fig_heat = go.Figure(go.Heatmap(
            z=trans_matrix,
            x=regime_labels, y=regime_labels,
            colorscale=[[0, BG_SECONDARY], [0.5, ACCENT_BLUE], [1, ACCENT_GREEN]],
            text=[[f"{cell:.2f}" for cell in row] for row in trans_matrix],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="P"),
        ))
        fig_heat.update_layout(
            title="Regime Transition Matrix",
            xaxis_title="To", yaxis_title="From",
            height=380,
        )
    else:
        fig_heat = empty_figure("No transition matrix available")

    return html.Div([
        _card_panel("Probability History",
                    dcc.Graph(figure=fig_area, config={"displayModeBar": False})),
        dbc.Row([
            dbc.Col(
                _card_panel("Current Probabilities",
                            dcc.Graph(figure=fig_bar, config={"displayModeBar": False})),
                md=5,
            ),
            dbc.Col(
                _card_panel("Transition Matrix",
                            dcc.Graph(figure=fig_heat, config={"displayModeBar": False})),
                md=7,
            ),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 3: Model Performance
# ---------------------------------------------------------------------------

def _render_model_tab(data: dict):
    """Render the model-performance tab with health metrics and retrain diagnostics."""
    cv_gap = data.get("cv_gap", 0.0)
    holdout_r2 = data.get("holdout_r2", 0.0)
    holdout_ic = data.get("holdout_ic", 0.0)
    ic_drift = data.get("ic_drift", 0.0)
    retrain_triggered = data.get("retrain_triggered", False)
    retrain_reasons = data.get("retrain_reasons", [])
    reg_hist = data.get("reg_hist_json", {})

    # ── Sub-metric cards ──────────────────────────────────────────────
    r2_color = ACCENT_GREEN if holdout_r2 > 0.02 else ACCENT_AMBER if holdout_r2 > 0 else ACCENT_RED
    ic_color = ACCENT_GREEN if holdout_ic > 0.05 else ACCENT_AMBER if holdout_ic > 0 else ACCENT_RED
    drift_color = ACCENT_GREEN if abs(ic_drift) < 0.05 else ACCENT_AMBER if abs(ic_drift) < 0.10 else ACCENT_RED
    gap_color = ACCENT_GREEN if cv_gap < 0.05 else ACCENT_AMBER if cv_gap < 0.15 else ACCENT_RED

    sub_cards = dbc.Row([
        dbc.Col(metric_card("Holdout R\u00b2", _fmt(holdout_r2, 4), color=r2_color), md=3),
        dbc.Col(metric_card("Holdout IC", _fmt(holdout_ic, 4), color=ic_color), md=3),
        dbc.Col(metric_card("IC Drift", _fmt(ic_drift, 4), color=drift_color,
                            subtitle="Recent vs baseline"), md=3),
        dbc.Col(metric_card("CV Gap", _fmt(cv_gap, 4), color=gap_color,
                            subtitle="IS-OOS degradation"), md=3),
    ], className="g-3 mb-3")

    # ── Model health timeline ─────────────────────────────────────────
    if reg_hist and "version_id" in reg_hist:
        versions = reg_hist["version_id"]
        fig_timeline = go.Figure()
        for metric_key, color, name in [
            ("cv_gap", ACCENT_AMBER, "CV Gap"),
            ("holdout_r2", ACCENT_BLUE, "Holdout R\u00b2"),
            ("holdout_spearman", ACCENT_GREEN, "Holdout IC"),
        ]:
            if metric_key in reg_hist:
                vals = reg_hist[metric_key]
                fig_timeline.add_trace(go.Scatter(
                    x=list(range(len(versions))),
                    y=vals,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                ))
        fig_timeline.update_layout(
            title="Model Registry Timeline",
            xaxis_title="Version",
            xaxis=dict(
                tickvals=list(range(len(versions))),
                ticktext=[str(v)[:8] for v in versions],
            ),
            yaxis_title="Metric Value",
            height=360,
            legend=dict(orientation="h", y=-0.2),
        )
    else:
        fig_timeline = empty_figure("No model registry history")

    # ── Retrain monitor text ──────────────────────────────────────────
    if retrain_triggered:
        status_line = "STATUS: RETRAIN TRIGGERED"
        status_color = ACCENT_RED
    else:
        status_line = "STATUS: Stable -- no retrain signals"
        status_color = ACCENT_GREEN

    retrain_lines = [status_line, ""]
    if retrain_reasons:
        retrain_lines.append("Trigger reasons:")
        for r in retrain_reasons:
            retrain_lines.append(f"  - {r}")
    else:
        retrain_lines.append("No active retrain triggers.")
    retrain_lines.append("")
    retrain_lines.append(f"Last check: {data.get('timestamp', '---')}")

    retrain_text = "\n".join(retrain_lines)

    return html.Div([
        sub_cards,
        dbc.Row([
            dbc.Col(
                _card_panel("Model Health Timeline",
                            dcc.Graph(figure=fig_timeline, config={"displayModeBar": False})),
                md=7,
            ),
            dbc.Col(
                _card_panel("Retrain Monitor", html.Pre(
                    retrain_text,
                    style={"color": status_color, "fontSize": "12px",
                           "fontFamily": "Menlo, monospace", "margin": "0",
                           "whiteSpace": "pre-wrap"},
                )),
                md=5,
            ),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 4: Feature Importance
# ---------------------------------------------------------------------------

def _render_features_tab(data: dict):
    """Render the feature-importance tab with global bars and regime heatmap."""
    imp_names = data.get("imp_names", [])
    imp_values = data.get("imp_values", [])
    rh_features = data.get("rh_features", [])
    rh_regimes = data.get("rh_regimes", [])
    rh_values = data.get("rh_values", [])

    # ── Global importance horizontal bar ──────────────────────────────
    if imp_names and imp_values:
        sorted_pairs = sorted(zip(imp_names, imp_values), key=lambda x: x[1])
        s_names = [p[0] for p in sorted_pairs]
        s_vals = [p[1] for p in sorted_pairs]
        fig_imp = go.Figure(go.Bar(
            y=s_names, x=s_vals,
            orientation="h",
            marker_color=ACCENT_BLUE,
            text=[f"{v:.4f}" for v in s_vals],
            textposition="auto",
        ))
        fig_imp.update_layout(
            title="Global Feature Importance (Top 20)",
            xaxis_title="Importance",
            height=max(400, len(s_names) * 24),
            margin=dict(l=160),
        )
    else:
        fig_imp = empty_figure("No feature importance data")

    # ── Regime heatmap ────────────────────────────────────────────────
    if rh_features and rh_regimes and rh_values:
        fig_rh = go.Figure(go.Heatmap(
            z=rh_values,
            x=rh_regimes,
            y=rh_features,
            colorscale=[[0, BG_SECONDARY], [0.5, ACCENT_AMBER], [1, ACCENT_GREEN]],
            text=[[f"{v:.3f}" for v in row] for row in rh_values],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Imp"),
        ))
        fig_rh.update_layout(
            title="Feature Importance by Regime",
            height=max(380, len(rh_features) * 28),
            margin=dict(l=160),
        )
    else:
        fig_rh = empty_figure("No regime-specific importance data")

    return html.Div([
        dbc.Row([
            dbc.Col(
                _card_panel("Global Feature Importance",
                            dcc.Graph(figure=fig_imp, config={"displayModeBar": False})),
                md=6,
            ),
            dbc.Col(
                _card_panel("Regime Feature Heatmap",
                            dcc.Graph(figure=fig_rh, config={"displayModeBar": False})),
                md=6,
            ),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 5: Trade Log
# ---------------------------------------------------------------------------

def _render_trades_tab(data: dict):
    """Render the trade-log tab with sortable/filterable backtest trade records."""
    trade_records = data.get("trade_records", [])

    column_map = {
        "ticker": "Ticker",
        "entry_date": "Entry",
        "exit_date": "Exit",
        "predicted_return": "Pred",
        "actual_return": "Actual",
        "net_return": "Net",
        "regime": "Regime",
        "confidence": "Confidence",
        "reason": "Reason",
    }

    if trade_records:
        available_cols = list(trade_records[0].keys())
        columns = []
        for key, label in column_map.items():
            if key in available_cols:
                col_spec = {"name": label, "id": key}
                if key in ("predicted_return", "actual_return", "net_return", "confidence"):
                    col_spec["type"] = "numeric"
                    col_spec["format"] = dash_table.Format.Format(
                        precision=4, scheme=dash_table.Format.Scheme.fixed
                    )
                columns.append(col_spec)

        table = dash_table.DataTable(
            data=trade_records,
            columns=columns,
            page_size=25,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1c2028",
                "color": TEXT_PRIMARY,
                "fontWeight": "600",
                "fontSize": "11px",
                "borderBottom": f"1px solid {BORDER}",
            },
            style_cell={
                "backgroundColor": BG_SECONDARY,
                "color": TEXT_SECONDARY,
                "fontSize": "11px",
                "fontFamily": "Menlo, monospace",
                "border": f"1px solid #21262d",
                "padding": "6px 10px",
                "textAlign": "left",
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
            style_filter={
                "backgroundColor": BG_PRIMARY,
                "color": TEXT_SECONDARY,
            },
        )
    else:
        table = html.Div(
            "No trade records available.",
            style={"color": TEXT_TERTIARY, "padding": "40px", "textAlign": "center"},
        )

    n_trades = data.get("n_trades", 0)
    subtitle = f"Showing last {min(200, n_trades)} of {n_trades} trades" if n_trades else "No trades"

    return _card_panel(
        f"Trade Log  |  {subtitle}",
        table,
    )


# ---------------------------------------------------------------------------
# Tab 6: Risk Metrics
# ---------------------------------------------------------------------------

def _render_risk_tab(data: dict):
    """Render the risk tab with return distribution, VaR, and risk summary metrics."""
    daily_returns = data.get("daily_returns_list", [])
    daily_dates = data.get("daily_returns_dates", [])
    var95 = data.get("var95", 0.0)
    cvar95 = data.get("cvar95", 0.0)
    var99 = data.get("var99", 0.0)
    cvar99 = data.get("cvar99", 0.0)
    annual_return = data.get("annual_return", 0.0)
    annual_vol = data.get("annual_vol", 0.0)
    max_dd = data.get("max_drawdown", 0.0)
    sharpe = data.get("sharpe", 0.0)
    sortino = data.get("sortino", 0.0)

    # ── Return distribution histogram with VaR lines ──────────────────
    if daily_returns:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=daily_returns,
            nbinsx=60,
            marker_color=ACCENT_BLUE,
            opacity=0.75,
            name="Daily Returns",
        ))
        # VaR 95 line
        fig_hist.add_vline(
            x=var95, line_dash="dash", line_color=ACCENT_AMBER, line_width=2,
            annotation_text=f"VaR 95: {var95:.3%}",
            annotation_position="top left",
            annotation_font_color=ACCENT_AMBER,
        )
        # VaR 99 line
        fig_hist.add_vline(
            x=var99, line_dash="dash", line_color=ACCENT_RED, line_width=2,
            annotation_text=f"VaR 99: {var99:.3%}",
            annotation_position="top left",
            annotation_font_color=ACCENT_RED,
        )
        fig_hist.update_layout(
            title="Return Distribution with Value-at-Risk",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            height=380,
            showlegend=False,
        )
    else:
        fig_hist = empty_figure("No return data for histogram")

    # ── Rolling risk chart (dual y-axis) ──────────────────────────────
    if daily_returns and daily_dates and len(daily_returns) > 20:
        returns_series = pd.Series(daily_returns, index=pd.to_datetime(daily_dates))
        rolling_vol = returns_series.rolling(20).std() * np.sqrt(252)
        rolling_sharpe = (returns_series.rolling(60).mean() * 252) / (returns_series.rolling(60).std() * np.sqrt(252))
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

        # Drawdown
        equity = (1 + returns_series).cumprod()
        peak = equity.cummax()
        drawdown = (equity / peak) - 1

        fig_roll = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
        )
        fig_roll.add_trace(go.Scatter(
            x=daily_dates, y=rolling_vol.tolist(),
            mode="lines", name="Rolling Vol (20D ann.)",
            line=dict(color=ACCENT_AMBER, width=1.5),
        ), secondary_y=False)
        fig_roll.add_trace(go.Scatter(
            x=daily_dates, y=drawdown.tolist(),
            mode="lines", name="Drawdown",
            line=dict(color=ACCENT_RED, width=1),
            fill="tozeroy",
            fillcolor="rgba(248, 81, 73, 0.15)",
        ), secondary_y=True)
        fig_roll.add_trace(go.Scatter(
            x=daily_dates, y=rolling_sharpe.tolist(),
            mode="lines", name="Rolling Sharpe (60D)",
            line=dict(color=ACCENT_GREEN, width=1, dash="dot"),
        ), secondary_y=False)

        fig_roll.update_layout(
            title="Rolling Risk Metrics",
            height=380,
            legend=dict(orientation="h", y=-0.2),
        )
        fig_roll.update_yaxes(title_text="Volatility / Sharpe", secondary_y=False)
        fig_roll.update_yaxes(title_text="Drawdown", secondary_y=True)
    else:
        fig_roll = empty_figure("Insufficient data for rolling risk chart")

    # ── Risk summary text ─────────────────────────────────────────────
    risk_lines = [
        "Risk Summary",
        "=" * 40,
        f"Annualized Return:   {annual_return:+.4f}  ({annual_return:.2%})",
        f"Annualized Vol:      {annual_vol:.4f}  ({annual_vol:.2%})",
        f"Sharpe Ratio:        {sharpe:.4f}",
        f"Sortino Ratio:       {sortino:.4f}",
        f"Max Drawdown:        {max_dd:.4f}  ({max_dd:.2%})",
        "",
        "Value-at-Risk",
        "-" * 40,
        f"VaR 95 (daily):      {var95:.4f}  ({var95:.2%})",
        f"CVaR 95 (daily):     {cvar95:.4f}  ({cvar95:.2%})",
        f"VaR 99 (daily):      {var99:.4f}  ({var99:.2%})",
        f"CVaR 99 (daily):     {cvar99:.4f}  ({cvar99:.2%})",
        "",
        f"Total trades:        {data.get('n_trades', 0)}",
        f"Daily observations:  {len(daily_returns)}",
    ]
    risk_text = "\n".join(risk_lines)

    return html.Div([
        dbc.Row([
            dbc.Col(
                _card_panel("Return Distribution",
                            dcc.Graph(figure=fig_hist, config={"displayModeBar": False})),
                md=7,
            ),
            dbc.Col(
                _card_panel("Risk Summary", html.Pre(
                    risk_text,
                    style={"color": TEXT_SECONDARY, "fontSize": "12px",
                           "fontFamily": "Menlo, monospace", "margin": "0",
                           "whiteSpace": "pre-wrap"},
                )),
                md=5,
            ),
        ], className="g-3"),
        _card_panel("Rolling Risk Metrics",
                    dcc.Graph(figure=fig_roll, config={"displayModeBar": False})),
    ])
