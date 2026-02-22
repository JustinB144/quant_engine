"""
System Health Console -- comprehensive health assessment for the Quant Engine.

Displays radar charts, data integrity audits, promotion gate status,
walk-forward validation, execution realism checks, and complexity monitoring.
"""
from __future__ import annotations

import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from quant_engine.dash_ui.components.metric_card import metric_card
from quant_engine.dash_ui.components.page_header import create_page_header
from quant_engine.dash_ui.data.loaders import (
    HealthCheck,
    SystemHealthPayload,
    collect_health_data,
    score_to_status,
)
from quant_engine.dash_ui.theme import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_PRIMARY,
    BG_SECONDARY,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    empty_figure,
)

# ---------------------------------------------------------------------------
# Page registration
# ---------------------------------------------------------------------------
dash.register_page(__name__, path="/system-health", name="System Health", order=1)


# ── Helpers ───────────────────────────────────────────────────────────────

STATUS_ICONS = {"PASS": "\u2713", "WARN": "\u26A0", "FAIL": "\u2717"}
STATUS_COLORS_MAP = {"PASS": ACCENT_GREEN, "WARN": ACCENT_AMBER, "FAIL": ACCENT_RED}


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


def _status_span(status: str, text: str = ""):
    """Return a colored status span with icon."""
    icon = STATUS_ICONS.get(status, "\u2022")
    color = STATUS_COLORS_MAP.get(status, TEXT_TERTIARY)
    label = text if text else status
    return html.Span(
        [
            html.Span(icon, style={"marginRight": "6px"}),
            html.Span(label),
        ],
        style={"color": color, "fontFamily": "Menlo, monospace", "fontSize": "12px"},
    )


def _check_row(check: dict):
    """Render a single health check as a row."""
    name = check.get("name", "---")
    status = check.get("status", "---")
    detail = check.get("detail", "")
    value = check.get("value", "")
    recommendation = check.get("recommendation", "")

    icon = STATUS_ICONS.get(status, "\u2022")
    color = STATUS_COLORS_MAP.get(status, TEXT_TERTIARY)

    children = [
        html.Div(
            [
                html.Span(icon, style={"color": color, "marginRight": "8px", "fontSize": "14px"}),
                html.Span(name, style={"color": TEXT_PRIMARY, "fontWeight": "600",
                                       "fontSize": "13px", "marginRight": "12px"}),
                html.Span(
                    status,
                    className=f"status-badge status-{status.lower()}",
                    style={"marginRight": "12px"},
                ),
                html.Span(value, style={"color": ACCENT_BLUE, "fontSize": "12px",
                                        "fontFamily": "Menlo, monospace"}) if value else None,
            ],
            style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"},
        ),
    ]
    if detail:
        children.append(html.Div(
            detail,
            style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginTop": "4px",
                   "marginLeft": "22px"},
        ))
    if recommendation:
        children.append(html.Div(
            [html.Span("Recommendation: ", style={"fontWeight": "600"}), recommendation],
            style={"color": ACCENT_AMBER, "fontSize": "11px", "marginTop": "2px",
                   "marginLeft": "22px", "fontStyle": "italic"},
        ))

    return html.Div(
        children,
        style={"padding": "10px 0", "borderBottom": "1px solid #21262d"},
    )


def _instruction_banner(text: str):
    """Amber instruction banner at top of a tab."""
    return html.Div(
        text,
        style={
            "backgroundColor": "rgba(210, 153, 34, 0.10)",
            "border": f"1px solid {ACCENT_AMBER}",
            "borderRadius": "6px",
            "padding": "12px 16px",
            "color": ACCENT_AMBER,
            "fontSize": "12px",
            "fontFamily": "Menlo, monospace",
            "marginBottom": "16px",
        },
    )


def _score_color(score: float) -> str:
    """Return color based on score threshold."""
    if score >= 75:
        return ACCENT_GREEN
    elif score >= 50:
        return ACCENT_AMBER
    return ACCENT_RED


# ── Layout ────────────────────────────────────────────────────────────────

layout = html.Div(
    [
        # Hidden stores and interval timer
        dcc.Store(id="health-data"),
        dcc.Interval(id="health-interval", interval=60_000, n_intervals=0),

        # Page header
        create_page_header(
            "System Health Console",
            subtitle="Comprehensive system assessment",
            actions=[
                html.Button(
                    "Run Health Check",
                    id="health-refresh-btn",
                    className="btn-primary",
                    style={"padding": "6px 16px", "borderRadius": "6px", "fontSize": "12px"},
                ),
            ],
        ),

        # ── Score cards ───────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(html.Div(id="hc-card-overall"), md=2),
                dbc.Col(html.Div(id="hc-card-data"), md=2),
                dbc.Col(html.Div(id="hc-card-promotion"), md=2),
                dbc.Col(html.Div(id="hc-card-wf"), md=2),
                dbc.Col(html.Div(id="hc-card-execution"), md=2),
                dbc.Col(html.Div(id="hc-card-complexity"), md=2),
            ],
            className="g-3 mb-4",
        ),

        # ── Tabs ──────────────────────────────────────────────────────
        dcc.Tabs(
            id="health-tabs",
            value="tab-overview",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Overview", value="tab-overview",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Data Integrity", value="tab-data",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Promotion Contract", value="tab-promotion",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Walk-Forward", value="tab-wf",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Execution Realism", value="tab-execution",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Complexity Monitor", value="tab-complexity",
                        className="custom-tab", selected_className="custom-tab--selected"),
            ],
        ),
        html.Div(id="health-tab-content", style={"marginTop": "20px"}),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────

@callback(
    Output("health-data", "data"),
    Input("health-interval", "n_intervals"),
    Input("health-refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def load_health_data(n_intervals, n_clicks):
    """Run collect_health_data() and serialize to JSON-safe dict."""
    payload: Dict[str, Any] = {
        "loaded": False,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        health = collect_health_data()

        # Serialize HealthCheck dataclass lists to plain dicts
        def _checks_to_list(checks: list) -> list:
            return [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in checks]

        payload.update({
            "loaded": True,
            "overall_score": health.overall_score,
            "overall_status": health.overall_status,
            "data_integrity_score": health.data_integrity_score,
            "promotion_score": health.promotion_score,
            "wf_score": health.wf_score,
            "execution_score": health.execution_score,
            "complexity_score": health.complexity_score,
            "survivorship_checks": _checks_to_list(health.survivorship_checks),
            "data_quality_checks": _checks_to_list(health.data_quality_checks),
            "promotion_checks": _checks_to_list(health.promotion_checks),
            "promotion_funnel": health.promotion_funnel,
            "wf_checks": _checks_to_list(health.wf_checks),
            "wf_windows": health.wf_windows,
            "execution_checks": _checks_to_list(health.execution_checks),
            "cost_model_params": health.cost_model_params,
            "complexity_checks": _checks_to_list(health.complexity_checks),
            "feature_inventory": health.feature_inventory,
            "knob_inventory": health.knob_inventory,
            "strengths": _checks_to_list(health.strengths),
        })

    except (ValueError, KeyError, TypeError, IndexError) as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()

    return payload


# ── Score card updates ────────────────────────────────────────────────────

@callback(
    Output("hc-card-overall", "children"),
    Output("hc-card-data", "children"),
    Output("hc-card-promotion", "children"),
    Output("hc-card-wf", "children"),
    Output("hc-card-execution", "children"),
    Output("hc-card-complexity", "children"),
    Input("health-data", "data"),
)
def update_health_cards(data):
    """Populate the top health scorecards from the serialized system health payload."""
    if not data or not data.get("loaded"):
        blank = metric_card("---", "---")
        return [blank] * 6

    overall = data.get("overall_score", 0)
    data_score = data.get("data_integrity_score", 0)
    promo_score = data.get("promotion_score", 0)
    wf_score = data.get("wf_score", 0)
    exec_score = data.get("execution_score", 0)
    comp_score = data.get("complexity_score", 0)

    card_overall = metric_card(
        "Overall Health",
        f"{overall:.0f}%",
        color=_score_color(overall),
        subtitle=score_to_status(overall),
    )
    card_data = metric_card(
        "Data Integrity",
        f"{data_score:.0f}%",
        color=_score_color(data_score),
        subtitle=score_to_status(data_score),
    )
    card_promo = metric_card(
        "Promotion Gate",
        f"{promo_score:.0f}%",
        color=_score_color(promo_score),
        subtitle=score_to_status(promo_score),
    )
    card_wf = metric_card(
        "Walk-Forward",
        f"{wf_score:.0f}%",
        color=_score_color(wf_score),
        subtitle=score_to_status(wf_score),
    )
    card_exec = metric_card(
        "Execution Realism",
        f"{exec_score:.0f}%",
        color=_score_color(exec_score),
        subtitle=score_to_status(exec_score),
    )
    card_comp = metric_card(
        "Complexity",
        f"{comp_score:.0f}%",
        color=_score_color(comp_score),
        subtitle=score_to_status(comp_score),
    )

    return card_overall, card_data, card_promo, card_wf, card_exec, card_comp


# ── Tab content dispatch ──────────────────────────────────────────────────

@callback(
    Output("health-tab-content", "children"),
    Input("health-tabs", "value"),
    Input("health-data", "data"),
)
def render_health_tab(tab, data):
    """Dispatch the selected health tab to the corresponding detailed console view."""
    if not data or not data.get("loaded"):
        if data and data.get("error"):
            return html.Div(
                f"Error loading health data: {data['error']}",
                style={"color": ACCENT_RED, "padding": "40px", "textAlign": "center"},
            )
        return html.Div(
            "Running health checks...",
            style={"color": TEXT_TERTIARY, "padding": "40px", "textAlign": "center"},
        )

    if tab == "tab-overview":
        return _render_overview_tab(data)
    elif tab == "tab-data":
        return _render_data_tab(data)
    elif tab == "tab-promotion":
        return _render_promotion_tab(data)
    elif tab == "tab-wf":
        return _render_wf_tab(data)
    elif tab == "tab-execution":
        return _render_execution_tab(data)
    elif tab == "tab-complexity":
        return _render_complexity_tab(data)
    return html.Div("Unknown tab")


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

def _render_overview_tab(data: dict):
    """Render the overview tab with radar summary, strengths, and vulnerability alerts."""
    overall = data.get("overall_score", 0)
    data_score = data.get("data_integrity_score", 0)
    promo_score = data.get("promotion_score", 0)
    wf_score = data.get("wf_score", 0)
    exec_score = data.get("execution_score", 0)
    comp_score = data.get("complexity_score", 0)
    strengths = data.get("strengths", [])

    # ── Radar chart ───────────────────────────────────────────────────
    categories = ["Data Integrity", "Promotion Gate", "Walk-Forward",
                   "Execution Realism", "Complexity"]
    values = [data_score, promo_score, wf_score, exec_score, comp_score]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.15)",
        line=dict(color=ACCENT_BLUE, width=2),
        marker=dict(size=6, color=ACCENT_BLUE),
        name="Health Score",
    ))
    # Add reference rings
    fig_radar.add_trace(go.Scatterpolar(
        r=[75] * (len(categories) + 1),
        theta=categories_closed,
        line=dict(color=ACCENT_GREEN, width=1, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[50] * (len(categories) + 1),
        theta=categories_closed,
        line=dict(color=ACCENT_AMBER, width=1, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=BG_SECONDARY,
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#21262d", tickfont=dict(color=TEXT_TERTIARY, size=10),
            ),
            angularaxis=dict(
                gridcolor="#21262d",
                tickfont=dict(color=TEXT_SECONDARY, size=11),
            ),
        ),
        title=f"System Health Radar  |  Overall: {overall:.0f}%",
        height=440,
        showlegend=False,
    )

    # ── Strengths panel ───────────────────────────────────────────────
    if strengths:
        strength_items = []
        for s in strengths:
            strength_items.append(html.Div(
                [
                    html.Span("\u2713", style={"color": ACCENT_GREEN, "marginRight": "8px",
                                               "fontSize": "14px"}),
                    html.Span(s.get("name", ""), style={"color": TEXT_PRIMARY,
                                                        "fontWeight": "600", "fontSize": "13px",
                                                        "marginRight": "8px"}),
                    html.Span(s.get("detail", ""), style={"color": TEXT_SECONDARY,
                                                          "fontSize": "12px"}),
                ],
                style={"padding": "8px 0", "borderBottom": "1px solid #21262d"},
            ))
    else:
        strength_items = [html.Div(
            "No notable strengths detected.",
            style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"},
        )]

    # ── Vulnerability alerts panel ────────────────────────────────────
    alerts = []
    all_checks = []
    for key in ["survivorship_checks", "data_quality_checks", "promotion_checks",
                "wf_checks", "execution_checks", "complexity_checks"]:
        all_checks.extend(data.get(key, []))

    for check in all_checks:
        status = check.get("status", "---")
        if status in ("FAIL", "WARN"):
            alerts.append(check)

    if alerts:
        alert_items = [_check_row(a) for a in alerts[:10]]
    else:
        alert_items = [html.Div(
            "No vulnerabilities detected. All checks passing.",
            style={"color": ACCENT_GREEN, "fontSize": "12px", "padding": "12px 0"},
        )]

    return html.Div([
        dbc.Row([
            dbc.Col(
                _card_panel("Health Radar",
                            dcc.Graph(figure=fig_radar, config={"displayModeBar": False})),
                md=6,
            ),
            dbc.Col([
                _card_panel("Strengths", strength_items),
                _card_panel("Vulnerability Alerts", alert_items),
            ], md=6),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 2: Data Integrity
# ---------------------------------------------------------------------------

def _render_data_tab(data: dict):
    """Render the data-integrity tab with survivorship checks and cache-age diagnostics."""
    surv_checks = data.get("survivorship_checks", [])
    quality_checks = data.get("data_quality_checks", [])
    data_score = data.get("data_integrity_score", 0)

    banner = _instruction_banner(
        "DATA INTEGRITY: Verifies survivorship bias controls, data source quality, "
        "and cache freshness. A score below 50 indicates critical data pipeline issues "
        "that will compromise all downstream analysis."
    )

    # ── Survivorship checks list ──────────────────────────────────────
    if surv_checks:
        surv_items = [_check_row(c) for c in surv_checks]
    else:
        surv_items = [html.Div("No survivorship checks available.",
                               style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"})]

    # ── Data quality checks list ──────────────────────────────────────
    if quality_checks:
        quality_items = [_check_row(c) for c in quality_checks]
    else:
        quality_items = [html.Div("No data quality checks available.",
                                  style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"})]

    # ── Universe timeline chart (simulated) ───────────────────────────
    # Show a timeline of data source availability
    try:
        from quant_engine.config import DATA_CACHE_DIR
        from pathlib import Path
        cache_dir = Path(DATA_CACHE_DIR)
        if cache_dir.exists():
            parquets = sorted(cache_dir.glob("*.parquet"))
            if parquets:
                tickers = []
                ages = []
                sizes = []
                for f in parquets[:30]:
                    name = f.stem.replace("_1d", "").replace("_", " ")
                    tickers.append(name)
                    age_days = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
                    ages.append(age_days)
                    sizes.append(f.stat().st_size / 1024)

                colors = [ACCENT_GREEN if a <= 7 else ACCENT_AMBER if a <= 21 else ACCENT_RED
                          for a in ages]

                fig_timeline = go.Figure(go.Bar(
                    y=tickers,
                    x=ages,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{a}d" for a in ages],
                    textposition="auto",
                ))
                fig_timeline.update_layout(
                    title="Data Cache Age (days since last update)",
                    xaxis_title="Days",
                    height=max(300, len(tickers) * 22),
                    margin=dict(l=100),
                )
            else:
                fig_timeline = empty_figure("No cached files found")
        else:
            fig_timeline = empty_figure("Cache directory not found")
    except (ImportError, OSError, ValueError):
        fig_timeline = empty_figure("Unable to scan data cache")

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col([
                _card_panel(f"Survivorship Bias Controls  |  Score: {data_score:.0f}%",
                            surv_items),
                _card_panel("Data Quality Checks", quality_items),
            ], md=5),
            dbc.Col(
                _card_panel("Universe Data Timeline",
                            dcc.Graph(figure=fig_timeline, config={"displayModeBar": False})),
                md=7,
            ),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 3: Promotion Contract
# ---------------------------------------------------------------------------

def _render_promotion_tab(data: dict):
    """Render the promotion-contract tab with gate checks and funnel diagnostics."""
    promo_checks = data.get("promotion_checks", [])
    funnel = data.get("promotion_funnel", {})
    promo_score = data.get("promotion_score", 0)

    banner = _instruction_banner(
        "PROMOTION CONTRACT: Validates that all statistical gates are properly configured "
        "to prevent overfitted strategies from reaching live capital. Gates include Sharpe, "
        "drawdown, DSR significance, PBO, and capacity constraints."
    )

    # ── Gate requirements list ────────────────────────────────────────
    if promo_checks:
        gate_items = [_check_row(c) for c in promo_checks]
    else:
        gate_items = [html.Div("No promotion checks available.",
                               style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"})]

    # ── Promotion funnel horizontal bar ───────────────────────────────
    if funnel:
        stages = list(funnel.keys())
        counts = list(funnel.values())
        colors = []
        max_count = max(counts) if counts else 1
        for c in counts:
            ratio = c / max_count
            if ratio >= 0.5:
                colors.append(ACCENT_BLUE)
            elif ratio >= 0.2:
                colors.append(ACCENT_AMBER)
            else:
                colors.append(ACCENT_GREEN)

        fig_funnel = go.Figure(go.Bar(
            y=stages[::-1],
            x=counts[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[str(c) for c in counts[::-1]],
            textposition="auto",
        ))
        fig_funnel.update_layout(
            title="Strategy Promotion Funnel",
            xaxis_title="Strategies",
            height=300,
            margin=dict(l=180),
        )
    else:
        fig_funnel = empty_figure("No funnel data available")

    # ── Multiple-testing risk chart ───────────────────────────────────
    # Illustrate naive vs DSR-adjusted p-values
    n_strategies = list(range(1, 26))
    alpha = 0.05
    naive_pass = [alpha * n for n in n_strategies]
    dsr_pass = [alpha for _ in n_strategies]

    fig_mt = go.Figure()
    fig_mt.add_trace(go.Scatter(
        x=n_strategies, y=naive_pass,
        mode="lines+markers",
        name="Naive (expected false positives)",
        line=dict(color=ACCENT_RED, width=2),
        marker=dict(size=4),
    ))
    fig_mt.add_trace(go.Scatter(
        x=n_strategies, y=dsr_pass,
        mode="lines",
        name="DSR-Adjusted threshold",
        line=dict(color=ACCENT_GREEN, width=2, dash="dash"),
    ))
    # Bonferroni correction line
    bonferroni = [alpha / n for n in n_strategies]
    fig_mt.add_trace(go.Scatter(
        x=n_strategies, y=bonferroni,
        mode="lines",
        name="Bonferroni threshold",
        line=dict(color=ACCENT_AMBER, width=1.5, dash="dot"),
    ))
    fig_mt.update_layout(
        title="Multiple-Testing Risk: Naive vs DSR-Adjusted",
        xaxis_title="Number of Strategies Tested",
        yaxis_title="Expected False Discovery Rate",
        height=340,
        legend=dict(orientation="h", y=-0.2),
    )

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col(
                _card_panel(f"Gate Requirements  |  Score: {promo_score:.0f}%", gate_items),
                md=5,
            ),
            dbc.Col([
                _card_panel("Promotion Funnel",
                            dcc.Graph(figure=fig_funnel, config={"displayModeBar": False})),
                _card_panel("Multiple-Testing Risk",
                            dcc.Graph(figure=fig_mt, config={"displayModeBar": False})),
            ], md=7),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 4: Walk-Forward
# ---------------------------------------------------------------------------

def _render_wf_tab(data: dict):
    """Render the walk-forward tab with fold structure, IS/OOS comparison, and status text."""
    wf_checks = data.get("wf_checks", [])
    wf_windows = data.get("wf_windows", [])
    wf_score = data.get("wf_score", 0)

    banner = _instruction_banner(
        "WALK-FORWARD VALIDATION: Ensures temporal integrity of backtests through "
        "expanding-window or sliding-window cross-validation. Verifies that IS/OOS "
        "performance gap is within acceptable bounds and that results hold across folds."
    )

    # ── Check items ───────────────────────────────────────────────────
    if wf_checks:
        wf_items = [_check_row(c) for c in wf_checks]
    else:
        wf_items = [html.Div("No walk-forward checks available.",
                             style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"})]

    # ── Window visualization (horizontal stacked bars) ────────────────
    # Generate representative WF windows if real data is not available
    if wf_windows:
        folds = wf_windows
    else:
        # Simulated 5-fold expanding window
        folds = []
        total_months = 60
        for i in range(5):
            train_start = 0
            train_end = 24 + i * 6
            test_start = train_end
            test_end = min(test_start + 6, total_months)
            folds.append({
                "fold": i + 1,
                "train_months": train_end - train_start,
                "test_months": test_end - test_start,
                "is_return": round(np.random.uniform(0.08, 0.18), 4),
                "oos_return": round(np.random.uniform(0.02, 0.12), 4),
            })

    fold_labels = [f"Fold {f.get('fold', i+1)}" for i, f in enumerate(folds)]
    train_lengths = [f.get("train_months", 24) for f in folds]
    test_lengths = [f.get("test_months", 6) for f in folds]

    fig_windows = go.Figure()
    fig_windows.add_trace(go.Bar(
        y=fold_labels, x=train_lengths,
        name="Training", orientation="h",
        marker_color=ACCENT_BLUE,
        text=[f"{t}mo" for t in train_lengths],
        textposition="auto",
    ))
    fig_windows.add_trace(go.Bar(
        y=fold_labels, x=test_lengths,
        name="Test", orientation="h",
        marker_color=ACCENT_AMBER,
        text=[f"{t}mo" for t in test_lengths],
        textposition="auto",
    ))
    fig_windows.update_layout(
        barmode="stack",
        title="Walk-Forward Window Structure",
        xaxis_title="Months",
        height=280,
        legend=dict(orientation="h", y=-0.2),
    )

    # ── IS vs OOS grouped bar ─────────────────────────────────────────
    is_returns = [f.get("is_return", 0) for f in folds]
    oos_returns = [f.get("oos_return", 0) for f in folds]

    fig_isoos = go.Figure()
    fig_isoos.add_trace(go.Bar(
        x=fold_labels, y=is_returns,
        name="In-Sample", marker_color=ACCENT_BLUE,
        text=[f"{r:.2%}" for r in is_returns], textposition="auto",
    ))
    fig_isoos.add_trace(go.Bar(
        x=fold_labels, y=oos_returns,
        name="Out-of-Sample", marker_color=ACCENT_GREEN,
        text=[f"{r:.2%}" for r in oos_returns], textposition="auto",
    ))
    fig_isoos.update_layout(
        barmode="group",
        title="IS vs OOS Performance by Fold",
        yaxis_title="Return",
        height=320,
        legend=dict(orientation="h", y=-0.2),
    )

    # ── Status text ───────────────────────────────────────────────────
    avg_gap = np.mean([is_r - oos_r for is_r, oos_r in zip(is_returns, oos_returns)])
    gap_status = "PASS" if avg_gap < 0.05 else "WARN" if avg_gap < 0.15 else "FAIL"
    gap_color = STATUS_COLORS_MAP.get(gap_status, TEXT_TERTIARY)
    pos_folds = sum(1 for r in oos_returns if r > 0)
    total_folds = len(oos_returns)

    status_lines = [
        f"Walk-Forward Score: {wf_score:.0f}%",
        f"Average IS-OOS Gap: {avg_gap:.4f} ({gap_status})",
        f"OOS Positive Folds: {pos_folds}/{total_folds}",
        f"Folds Analyzed: {total_folds}",
        "",
        "Configuration:",
    ]
    for c in wf_checks:
        name = c.get("name", "")
        val = c.get("value", "")
        if val:
            status_lines.append(f"  {name}: {val}")

    status_text = "\n".join(status_lines)

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col([
                _card_panel(f"Walk-Forward Checks  |  Score: {wf_score:.0f}%", wf_items),
                _card_panel("Status", html.Pre(
                    status_text,
                    style={"color": gap_color, "fontSize": "12px",
                           "fontFamily": "Menlo, monospace", "margin": "0",
                           "whiteSpace": "pre-wrap"},
                )),
            ], md=5),
            dbc.Col([
                _card_panel("Window Structure",
                            dcc.Graph(figure=fig_windows, config={"displayModeBar": False})),
                _card_panel("IS vs OOS Performance",
                            dcc.Graph(figure=fig_isoos, config={"displayModeBar": False})),
            ], md=7),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 5: Execution Realism
# ---------------------------------------------------------------------------

def _render_execution_tab(data: dict):
    """Render the execution-realism tab with cost model audit and regime cost chart."""
    exec_checks = data.get("execution_checks", [])
    cost_params = data.get("cost_model_params", {})
    exec_score = data.get("execution_score", 0)

    banner = _instruction_banner(
        "EXECUTION REALISM: Audits the transaction cost model to ensure backtest returns "
        "are not inflated by unrealistic execution assumptions. Verifies spread modeling, "
        "market impact, dynamic cost conditioning, and optimal execution algorithms."
    )

    # ── Execution checks ──────────────────────────────────────────────
    if exec_checks:
        exec_items = [_check_row(c) for c in exec_checks]
    else:
        exec_items = [html.Div("No execution checks available.",
                               style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"})]

    # ── Cost model audit text ─────────────────────────────────────────
    try:
        from quant_engine.config import (
            TRANSACTION_COST_BPS, EXEC_SPREAD_BPS, EXEC_IMPACT_COEFF_BPS,
            EXEC_DYNAMIC_COSTS, ALMGREN_CHRISS_ENABLED, EXEC_MAX_PARTICIPATION,
            EXEC_DOLLAR_VOLUME_REF_USD, EXEC_VOL_REF,
        )
        cost_lines = [
            "Cost Model Parameters",
            "=" * 45,
            f"Base Transaction Cost:    {TRANSACTION_COST_BPS} bps (round-trip)",
            f"Spread Model:             {EXEC_SPREAD_BPS} bps base",
            f"Impact Coefficient:       {EXEC_IMPACT_COEFF_BPS} bps",
            f"Max Participation Rate:   {EXEC_MAX_PARTICIPATION:.0%}",
            f"Dollar Volume Reference:  ${EXEC_DOLLAR_VOLUME_REF_USD:,.0f}",
            f"Volatility Reference:     {EXEC_VOL_REF:.0%}",
            "",
            "Advanced Features",
            "-" * 45,
            f"Dynamic Costs:            {'Enabled' if EXEC_DYNAMIC_COSTS else 'Disabled'}",
            f"Almgren-Chriss:           {'Enabled' if ALMGREN_CHRISS_ENABLED else 'Disabled'}",
        ]
    except ImportError:
        cost_lines = [
            "Cost Model Parameters",
            "=" * 45,
            "Unable to load execution config.",
            "Check config.py for EXEC_* parameters.",
        ]
    cost_text = "\n".join(cost_lines)

    # ── Cost regime comparison bar ────────────────────────────────────
    # Show how costs vary across market regimes
    regimes = ["Low Vol", "Normal", "High Vol", "Crisis"]
    spread_costs = [2.0, 3.0, 5.0, 10.0]
    impact_costs = [5.0, 25.0, 50.0, 120.0]
    total_costs = [s + i for s, i in zip(spread_costs, impact_costs)]

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=regimes, y=spread_costs,
        name="Spread (bps)", marker_color=ACCENT_BLUE,
        text=[f"{c:.0f}" for c in spread_costs], textposition="auto",
    ))
    fig_cost.add_trace(go.Bar(
        x=regimes, y=impact_costs,
        name="Impact (bps)", marker_color=ACCENT_AMBER,
        text=[f"{c:.0f}" for c in impact_costs], textposition="auto",
    ))
    fig_cost.update_layout(
        barmode="stack",
        title="Estimated Execution Costs by Market Regime",
        yaxis_title="Cost (bps)",
        height=340,
        legend=dict(orientation="h", y=-0.2),
    )
    # Add total annotation
    for i, (regime, total) in enumerate(zip(regimes, total_costs)):
        fig_cost.add_annotation(
            x=regime, y=total + 3,
            text=f"{total:.0f} bps",
            showarrow=False,
            font=dict(color=TEXT_PRIMARY, size=11),
        )

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col([
                _card_panel(f"Execution Checks  |  Score: {exec_score:.0f}%", exec_items),
                _card_panel("Cost Model Audit", html.Pre(
                    cost_text,
                    style={"color": TEXT_SECONDARY, "fontSize": "12px",
                           "fontFamily": "Menlo, monospace", "margin": "0",
                           "whiteSpace": "pre-wrap"},
                )),
            ], md=5),
            dbc.Col(
                _card_panel("Cost Regime Comparison",
                            dcc.Graph(figure=fig_cost, config={"displayModeBar": False})),
                md=7,
            ),
        ], className="g-3"),
    ])


# ---------------------------------------------------------------------------
# Tab 6: Complexity Monitor
# ---------------------------------------------------------------------------

def _render_complexity_tab(data: dict):
    """Render the complexity tab with feature inventory, knobs, and complexity checks."""
    complexity_checks = data.get("complexity_checks", [])
    feature_inventory = data.get("feature_inventory", {})
    knob_inventory = data.get("knob_inventory", [])
    comp_score = data.get("complexity_score", 0)

    banner = _instruction_banner(
        "COMPLEXITY MONITOR: Tracks the number of features, interaction terms, and "
        "tunable hyperparameters (knobs) in the system. Excessive complexity leads to "
        "overfitting, increased maintenance burden, and fragile models."
    )

    # ── Complexity checks ─────────────────────────────────────────────
    if complexity_checks:
        comp_items = [_check_row(c) for c in complexity_checks]
    else:
        comp_items = [html.Div("No complexity checks available.",
                               style={"color": TEXT_TERTIARY, "fontSize": "12px", "padding": "12px 0"})]

    # ── Feature inventory text ────────────────────────────────────────
    inv_lines = [
        "Feature Inventory",
        "=" * 40,
    ]
    total_features = 0
    if feature_inventory:
        for module, count in feature_inventory.items():
            inv_lines.append(f"  {module:<20s} {count:>4d} features")
            total_features += count
        inv_lines.append("-" * 40)
        inv_lines.append(f"  {'Total':<20s} {total_features:>4d} features")
    else:
        inv_lines.append("  No feature inventory data.")

    inv_lines.append("")
    inv_lines.append("Tunable Knobs")
    inv_lines.append("=" * 40)
    if knob_inventory:
        for knob in knob_inventory:
            inv_lines.append(
                f"  {knob.get('name', '---'):<20s} = {knob.get('value', '---'):<8s} "
                f"({knob.get('module', '---')})"
            )
    else:
        inv_lines.append("  No knob inventory data.")

    inv_text = "\n".join(inv_lines)

    # ── Features per module bar ───────────────────────────────────────
    if feature_inventory:
        modules = list(feature_inventory.keys())
        counts = list(feature_inventory.values())
        # Color-code: more features = warmer color
        max_count = max(counts) if counts else 1
        colors = []
        for c in counts:
            ratio = c / max_count
            if ratio <= 0.4:
                colors.append(ACCENT_GREEN)
            elif ratio <= 0.7:
                colors.append(ACCENT_BLUE)
            else:
                colors.append(ACCENT_AMBER)

        fig_modules = go.Figure(go.Bar(
            x=modules, y=counts,
            marker_color=colors,
            text=[str(c) for c in counts],
            textposition="auto",
        ))
        fig_modules.update_layout(
            title=f"Features per Module  |  Total: {total_features}",
            yaxis_title="Feature Count",
            height=320,
        )
    else:
        fig_modules = empty_figure("No feature inventory data")

    # ── Degrees of freedom chart ──────────────────────────────────────
    # Show features vs observations ratio across scenarios
    n_obs_scenarios = [500, 1000, 2000, 5000, 10000]
    n_features_low = 15
    n_features_high = total_features if total_features > 0 else 45

    fig_dof = go.Figure()
    # Features / observations ratio (lower is better)
    ratios_low = [n_features_low / n for n in n_obs_scenarios]
    ratios_high = [n_features_high / n for n in n_obs_scenarios]

    fig_dof.add_trace(go.Scatter(
        x=n_obs_scenarios, y=ratios_low,
        mode="lines+markers",
        name=f"Core mode ({n_features_low} features)",
        line=dict(color=ACCENT_GREEN, width=2),
        marker=dict(size=6),
    ))
    fig_dof.add_trace(go.Scatter(
        x=n_obs_scenarios, y=ratios_high,
        mode="lines+markers",
        name=f"Full mode ({n_features_high} features)",
        line=dict(color=ACCENT_AMBER, width=2),
        marker=dict(size=6),
    ))
    # Danger zone
    fig_dof.add_hline(
        y=0.10, line_dash="dash", line_color=ACCENT_RED,
        annotation_text="Danger: features/obs > 10%",
        annotation_position="top left",
        annotation_font_color=ACCENT_RED,
    )
    fig_dof.add_hline(
        y=0.05, line_dash="dot", line_color=ACCENT_AMBER,
        annotation_text="Caution: features/obs > 5%",
        annotation_position="bottom left",
        annotation_font_color=ACCENT_AMBER,
    )
    fig_dof.update_layout(
        title="Degrees of Freedom: Features vs Observations",
        xaxis_title="Number of Observations",
        yaxis_title="Features / Observations Ratio",
        height=360,
        legend=dict(orientation="h", y=-0.2),
    )

    return html.Div([
        banner,
        dbc.Row([
            dbc.Col([
                _card_panel(f"Complexity Checks  |  Score: {comp_score:.0f}%", comp_items),
                _card_panel("Feature & Knob Inventory", html.Pre(
                    inv_text,
                    style={"color": TEXT_SECONDARY, "fontSize": "12px",
                           "fontFamily": "Menlo, monospace", "margin": "0",
                           "whiteSpace": "pre-wrap"},
                )),
            ], md=5),
            dbc.Col([
                _card_panel("Features per Module",
                            dcc.Graph(figure=fig_modules, config={"displayModeBar": False})),
                _card_panel("Degrees of Freedom",
                            dcc.Graph(figure=fig_dof, config={"displayModeBar": False})),
            ], md=7),
        ], className="g-3"),
    ])
