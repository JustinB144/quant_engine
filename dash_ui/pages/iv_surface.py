"""IV Surface Lab -- SVI, Heston, and Arb-Aware volatility surface modeling."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

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
)

dash.register_page(__name__, path="/iv-surface", name="IV Surface", order=6)

# ---------------------------------------------------------------------------
# SVI Presets
# ---------------------------------------------------------------------------

SVI_PRESETS = {
    "Equity Normal": {"a": 0.04, "b": 0.35, "rho": -0.40, "m": 0.00, "sigma": 0.25},
    "Equity Stressed": {"a": 0.08, "b": 0.55, "rho": -0.65, "m": -0.05, "sigma": 0.15},
    "Commodity Smile": {"a": 0.03, "b": 0.30, "rho": 0.10, "m": 0.00, "sigma": 0.35},
    "Flat Vol": {"a": 0.06, "b": 0.08, "rho": -0.10, "m": 0.00, "sigma": 0.40},
    "Steep Skew": {"a": 0.03, "b": 0.50, "rho": -0.80, "m": -0.08, "sigma": 0.12},
}

HESTON_PRESETS = {
    "Normal Market": {"v0": 0.04, "theta": 0.06, "kappa": 2.0, "sigma": 0.40, "rho": -0.70},
    "Vol Storm": {"v0": 0.15, "theta": 0.08, "kappa": 1.0, "sigma": 0.80, "rho": -0.80},
    "Low Vol": {"v0": 0.02, "theta": 0.03, "kappa": 3.0, "sigma": 0.25, "rho": -0.50},
}

# Surface grid parameters
S = 100.0
R = 0.05
MONEYNESS = np.linspace(0.65, 1.35, 35)
STRIKES = S * MONEYNESS
EXPIRIES = np.linspace(0.08, 2.0, 25)

# Smile expiries for 2D curves
SMILE_EXPIRIES = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}
SMILE_COLORS = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_RED]

# ---------------------------------------------------------------------------
# SVI computation (analytical, vectorized)
# ---------------------------------------------------------------------------


def compute_svi_surface(a, b, rho, m, sigma):
    """Compute SVI implied volatility surface analytically."""
    F_values = S * np.exp(R * EXPIRIES)
    n_K = len(STRIKES)
    n_T = len(EXPIRIES)
    iv_grid = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(EXPIRIES):
        F = F_values[i]
        k = np.log(STRIKES / F)
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        w = np.maximum(w, 1e-10)
        iv_grid[i, :] = np.sqrt(w / T)

    K_grid, T_grid = np.meshgrid(MONEYNESS, EXPIRIES)
    return K_grid, T_grid, iv_grid


def compute_svi_smiles(a, b, rho, m, sigma):
    """Compute individual smile curves for select expiries."""
    smiles = {}
    for label, T in SMILE_EXPIRIES.items():
        F = S * np.exp(R * T)
        k = np.log(STRIKES / F)
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
        w = np.maximum(w, 1e-10)
        iv = np.sqrt(w / T)
        smiles[label] = iv
    return smiles


# ---------------------------------------------------------------------------
# 3D camera defaults
# ---------------------------------------------------------------------------

SURFACE_CAMERA = dict(
    eye=dict(x=1.6, y=-1.6, z=0.8),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=-0.1),
)

# ---------------------------------------------------------------------------
# Slider helpers
# ---------------------------------------------------------------------------

_slider_style = {"marginBottom": "20px"}
_label_style = {"fontSize": "11px", "color": TEXT_TERTIARY, "marginBottom": "2px"}


def _svi_slider(param_id, label, min_val, max_val, step, value, marks=None):
    """Create a consistently styled labeled slider for SVI/Heston parameter controls."""
    if marks is None:
        marks = {min_val: str(min_val), max_val: str(max_val)}
    return html.Div(
        [
            html.Label(label, style=_label_style),
            dcc.Slider(
                id=param_id,
                min=min_val,
                max=max_val,
                step=step,
                value=value,
                marks={k: {"label": str(v), "style": {"color": TEXT_TERTIARY, "fontSize": "10px"}} for k, v in marks.items()},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style=_slider_style,
    )


# ---------------------------------------------------------------------------
# Tab 1: SVI Surface
# ---------------------------------------------------------------------------

svi_controls = html.Div(
    [
        html.Div("SVI Parameters", className="card-panel-header"),
        html.Label("Preset", style=_label_style),
        dcc.Dropdown(
            id="svi-preset-dd",
            options=[{"label": k, "value": k} for k in SVI_PRESETS],
            value="Equity Normal",
            clearable=False,
            style={"marginBottom": "16px", "fontSize": "12px"},
        ),
        _svi_slider("svi-a", "a (level)", -0.10, 0.20, 0.005, 0.04, {-0.10: "-0.10", 0.0: "0", 0.20: "0.20"}),
        _svi_slider("svi-b", "b (angle)", 0.01, 1.00, 0.01, 0.35, {0.01: "0.01", 0.50: "0.50", 1.00: "1.00"}),
        _svi_slider("svi-rho", "rho (skew)", -0.95, 0.95, 0.05, -0.40, {-0.95: "-0.95", 0.0: "0", 0.95: "0.95"}),
        _svi_slider("svi-m", "m (shift)", -0.30, 0.30, 0.01, 0.00, {-0.30: "-0.30", 0.0: "0", 0.30: "0.30"}),
        _svi_slider("svi-sigma", "sigma (curvature)", 0.01, 0.80, 0.01, 0.25, {0.01: "0.01", 0.40: "0.40", 0.80: "0.80"}),
    ],
    className="card-panel",
)

tab_svi = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(svi_controls, md=3, lg=2),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id="svi-surface-3d", figure=empty_figure("Adjusting SVI parameters..."))),
                        dcc.Loading(dcc.Graph(id="svi-smiles-2d", figure=empty_figure("Smile curves"))),
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
# Tab 2: Heston Surface
# ---------------------------------------------------------------------------

heston_controls = html.Div(
    [
        html.Div("Heston Parameters", className="card-panel-header"),
        html.Label("Preset", style=_label_style),
        dcc.Dropdown(
            id="heston-preset-dd",
            options=[{"label": k, "value": k} for k in HESTON_PRESETS],
            value="Normal Market",
            clearable=False,
            style={"marginBottom": "16px", "fontSize": "12px"},
        ),
        _svi_slider("heston-v0", "v0 (init var)", 0.005, 0.30, 0.005, 0.04, {0.005: "0.005", 0.15: "0.15", 0.30: "0.30"}),
        _svi_slider("heston-theta", "theta (long var)", 0.005, 0.20, 0.005, 0.06, {0.005: "0.005", 0.10: "0.10", 0.20: "0.20"}),
        _svi_slider("heston-kappa", "kappa (reversion)", 0.10, 5.00, 0.10, 2.0, {0.10: "0.10", 2.50: "2.50", 5.00: "5.00"}),
        _svi_slider("heston-sigma", "sigma (vol-of-vol)", 0.05, 1.50, 0.05, 0.40, {0.05: "0.05", 0.75: "0.75", 1.50: "1.50"}),
        _svi_slider("heston-rho", "rho (correlation)", -0.95, 0.50, 0.05, -0.70, {-0.95: "-0.95", 0.0: "0", 0.50: "0.50"}),
        dbc.Button(
            "Compute Surface",
            id="heston-compute-btn",
            className="btn-primary",
            size="sm",
            style={"width": "100%", "marginTop": "8px"},
        ),
    ],
    className="card-panel",
)

tab_heston = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(heston_controls, md=3, lg=2),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id="heston-surface-3d", figure=empty_figure("Click Compute Surface")), type="circle"),
                        dcc.Loading(dcc.Graph(id="heston-smiles-2d", figure=empty_figure("Heston smile curves"))),
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
# Tab 3: Arb-Aware SVI
# ---------------------------------------------------------------------------

arb_controls = html.Div(
    [
        html.Div("Arb-Aware SVI Builder", className="card-panel-header"),
        html.Label("Spot Price", style=_label_style),
        dbc.Input(id="arb-spot", type="number", value=100.0, min=1, step=1, size="sm",
                  style={"marginBottom": "10px", "backgroundColor": BG_PRIMARY,
                         "borderColor": BORDER, "color": TEXT_SECONDARY}),
        html.Label("Risk-Free Rate", style=_label_style),
        dbc.Input(id="arb-rate", type="number", value=0.05, min=0.0, max=0.20, step=0.005, size="sm",
                  style={"marginBottom": "10px", "backgroundColor": BG_PRIMARY,
                         "borderColor": BORDER, "color": TEXT_SECONDARY}),
        html.Label("Dividend Yield", style=_label_style),
        dbc.Input(id="arb-div", type="number", value=0.01, min=0.0, max=0.10, step=0.005, size="sm",
                  style={"marginBottom": "10px", "backgroundColor": BG_PRIMARY,
                         "borderColor": BORDER, "color": TEXT_SECONDARY}),
        html.Label("Noise Level", style=_label_style),
        dbc.Input(id="arb-noise", type="number", value=0.003, min=0.0, max=0.05, step=0.001, size="sm",
                  style={"marginBottom": "10px", "backgroundColor": BG_PRIMARY,
                         "borderColor": BORDER, "color": TEXT_SECONDARY}),
        html.Label("Max Iterations", style=_label_style),
        dbc.Input(id="arb-max-iter", type="number", value=300, min=50, max=1000, step=50, size="sm",
                  style={"marginBottom": "14px", "backgroundColor": BG_PRIMARY,
                         "borderColor": BORDER, "color": TEXT_SECONDARY}),
        dbc.Button(
            "Build Surface",
            id="arb-build-btn",
            className="btn-primary",
            size="sm",
            style={"width": "100%"},
        ),
    ],
    className="card-panel",
)

tab_arb = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(arb_controls, md=3, lg=2),
                dbc.Col(
                    [
                        dcc.Loading(dcc.Graph(id="arb-surface-3d", figure=empty_figure("Click Build Surface")), type="circle"),
                        dcc.Loading(dcc.Graph(id="arb-smiles-2d", figure=empty_figure("Arb-free smile curves"))),
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
# Page Layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        create_page_header("IV Surface", subtitle="Implied volatility modeling and arbitrage"),
        dcc.Tabs(
            [
                dcc.Tab(tab_svi, label="SVI Surface", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(tab_heston, label="Heston Surface", className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(tab_arb, label="Arb-Aware SVI", className="custom-tab", selected_className="custom-tab--selected"),
            ],
            className="custom-tabs",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Helper: build 3D surface figure
# ---------------------------------------------------------------------------


def _build_surface_figure(K_grid, T_grid, iv_grid, title="IV Surface"):
    """Create a go.Surface figure with Inferno colorscale."""
    fig = go.Figure(data=[
        go.Surface(
            x=K_grid,
            y=T_grid,
            z=iv_grid * 100,  # Display as percentage
            colorscale="Inferno",
            colorbar=dict(
                title=dict(text="IV (%)", font=dict(color=TEXT_SECONDARY, size=11)),
                tickfont=dict(color=TEXT_SECONDARY, size=10),
                len=0.6,
            ),
            hovertemplate=(
                "Moneyness: %{x:.2f}<br>"
                "Expiry: %{y:.2f}Y<br>"
                "IV: %{z:.1f}%<extra></extra>"
            ),
        )
    ])
    fig.update_layout(
        title=title,
        height=520,
        scene=dict(
            xaxis=dict(title="Moneyness (K/S)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
            yaxis=dict(title="Expiry (Years)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
            zaxis=dict(title="IV (%)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
            bgcolor=BG_PRIMARY,
            camera=SURFACE_CAMERA,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _build_smiles_figure(smiles, title="Volatility Smiles"):
    """Create 2D smile curves for select expiries."""
    fig = go.Figure()
    for (label, ivs), color in zip(smiles.items(), SMILE_COLORS):
        fig.add_trace(go.Scatter(
            x=MONEYNESS,
            y=ivs * 100,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        title=title,
        height=320,
        xaxis=dict(title="Moneyness (K/S)"),
        yaxis=dict(title="IV (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# SVI preset callback -- sets slider values
# ---------------------------------------------------------------------------

@callback(
    Output("svi-a", "value"),
    Output("svi-b", "value"),
    Output("svi-rho", "value"),
    Output("svi-m", "value"),
    Output("svi-sigma", "value"),
    Input("svi-preset-dd", "value"),
    prevent_initial_call=True,
)
def set_svi_preset(preset_name):
    """Set slider values from preset selection."""
    p = SVI_PRESETS.get(preset_name, SVI_PRESETS["Equity Normal"])
    return p["a"], p["b"], p["rho"], p["m"], p["sigma"]


# ---------------------------------------------------------------------------
# SVI surface callback -- recompute on any slider change (instant)
# ---------------------------------------------------------------------------

@callback(
    Output("svi-surface-3d", "figure"),
    Output("svi-smiles-2d", "figure"),
    Input("svi-a", "value"),
    Input("svi-b", "value"),
    Input("svi-rho", "value"),
    Input("svi-m", "value"),
    Input("svi-sigma", "value"),
)
def update_svi_surface(a, b, rho, m, sigma):
    """Recompute SVI surface from slider values (analytical, fast)."""
    if any(v is None for v in [a, b, rho, m, sigma]):
        return empty_figure("Set all parameters"), empty_figure("Set all parameters")

    try:
        a, b, rho, m, sigma = float(a), float(b), float(rho), float(m), float(sigma)
    except (ValueError, TypeError):
        return empty_figure("Invalid parameter values"), empty_figure("Invalid parameter values")

    # Compute 3D surface
    K_grid, T_grid, iv_grid = compute_svi_surface(a, b, rho, m, sigma)
    fig_3d = _build_surface_figure(K_grid, T_grid, iv_grid, title="SVI Implied Volatility Surface")

    # Compute 2D smiles
    smiles = compute_svi_smiles(a, b, rho, m, sigma)
    fig_2d = _build_smiles_figure(smiles, title="SVI Smile Curves by Expiry")

    return fig_3d, fig_2d


# ---------------------------------------------------------------------------
# Heston preset callback
# ---------------------------------------------------------------------------

@callback(
    Output("heston-v0", "value"),
    Output("heston-theta", "value"),
    Output("heston-kappa", "value"),
    Output("heston-sigma", "value"),
    Output("heston-rho", "value"),
    Input("heston-preset-dd", "value"),
    prevent_initial_call=True,
)
def set_heston_preset(preset_name):
    """Set Heston slider values from preset."""
    p = HESTON_PRESETS.get(preset_name, HESTON_PRESETS["Normal Market"])
    return p["v0"], p["theta"], p["kappa"], p["sigma"], p["rho"]


# ---------------------------------------------------------------------------
# Heston surface callback -- triggered by Compute button
# ---------------------------------------------------------------------------

@callback(
    Output("heston-surface-3d", "figure"),
    Output("heston-smiles-2d", "figure"),
    Input("heston-compute-btn", "n_clicks"),
    State("heston-v0", "value"),
    State("heston-theta", "value"),
    State("heston-kappa", "value"),
    State("heston-sigma", "value"),
    State("heston-rho", "value"),
    prevent_initial_call=True,
)
def compute_heston_surface(n_clicks, v0, theta, kappa, sigma, rho):
    """Compute Heston IV surface (computationally intensive)."""
    if any(v is None for v in [v0, theta, kappa, sigma, rho]):
        return empty_figure("Set all Heston parameters"), empty_figure("Set all parameters")

    try:
        from quant_engine.models.iv.models import HestonModel, HestonParams
    except ImportError:
        msg = "HestonModel not available. Ensure models.iv.models is installed."
        return empty_figure(msg), empty_figure(msg)

    try:
        v0, theta, kappa, sigma, rho = float(v0), float(theta), float(kappa), float(sigma), float(rho)
        params = HestonParams(v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho)
        model = HestonModel(params)

        # Use fewer points for Heston (it's slow per-point)
        heston_moneyness = np.linspace(0.75, 1.25, 20)
        heston_strikes = S * heston_moneyness
        heston_expiries = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

        K_grid, T_grid, iv_grid = model.iv_surface(S, heston_strikes, heston_expiries, r=R)

        # Build 3D
        K_norm = K_grid / S  # Convert to moneyness for display
        fig_3d = go.Figure(data=[
            go.Surface(
                x=K_norm,
                y=T_grid,
                z=iv_grid * 100,
                colorscale="Inferno",
                colorbar=dict(
                    title=dict(text="IV (%)", font=dict(color=TEXT_SECONDARY, size=11)),
                    tickfont=dict(color=TEXT_SECONDARY, size=10),
                    len=0.6,
                ),
                hovertemplate=(
                    "Moneyness: %{x:.2f}<br>"
                    "Expiry: %{y:.2f}Y<br>"
                    "IV: %{z:.1f}%<extra></extra>"
                ),
            )
        ])
        fig_3d.update_layout(
            title="Heston Implied Volatility Surface",
            height=520,
            scene=dict(
                xaxis=dict(title="Moneyness (K/S)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
                yaxis=dict(title="Expiry (Years)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
                zaxis=dict(title="IV (%)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
                bgcolor=BG_PRIMARY,
                camera=SURFACE_CAMERA,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        # Build 2D smiles
        smile_expiries_list = [0.083, 0.25, 0.5, 1.0, 2.0]
        smile_labels = ["1M", "3M", "6M", "1Y", "2Y"]
        fig_2d = go.Figure()
        for T_val, label, color in zip(smile_expiries_list, smile_labels, SMILE_COLORS):
            ivs = []
            for K_val in heston_strikes:
                try:
                    iv = model.implied_vol(S, K_val, T_val, R)
                    ivs.append(iv * 100 if np.isfinite(iv) else np.nan)
                except (ValueError, TypeError, IndexError):
                    ivs.append(np.nan)
            fig_2d.add_trace(go.Scatter(
                x=heston_moneyness,
                y=ivs,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))
        fig_2d.update_layout(
            title="Heston Smile Curves by Expiry",
            height=320,
            xaxis=dict(title="Moneyness (K/S)"),
            yaxis=dict(title="IV (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Check Feller condition
        feller_ok = params.validate()
        if not feller_ok:
            fig_3d.add_annotation(
                text="Warning: Feller condition violated (2*kappa*theta < sigma^2)",
                xref="paper", yref="paper",
                x=0.5, y=0.0,
                showarrow=False,
                font=dict(color=ACCENT_AMBER, size=12),
                bgcolor=BG_TERTIARY,
                bordercolor=ACCENT_AMBER,
                borderwidth=1,
            )

        return fig_3d, fig_2d

    except (ValueError, KeyError, TypeError, IndexError) as e:
        err_msg = f"Heston computation error: {str(e)[:100]}"
        return empty_figure(err_msg), empty_figure(err_msg)


# ---------------------------------------------------------------------------
# Arb-Aware SVI callback
# ---------------------------------------------------------------------------

@callback(
    Output("arb-surface-3d", "figure"),
    Output("arb-smiles-2d", "figure"),
    Input("arb-build-btn", "n_clicks"),
    State("arb-spot", "value"),
    State("arb-rate", "value"),
    State("arb-div", "value"),
    State("arb-noise", "value"),
    State("arb-max-iter", "value"),
    prevent_initial_call=True,
)
def build_arb_free_surface(n_clicks, spot, rate, div_yield, noise, max_iter):
    """Build arbitrage-free SVI surface from synthetic market data."""
    try:
        from quant_engine.models.iv.models import ArbitrageFreeSVIBuilder, generate_synthetic_market_surface
    except ImportError:
        msg = "ArbitrageFreeSVIBuilder not available. Ensure models.iv.models is installed."
        return empty_figure(msg), empty_figure(msg)

    try:
        spot = float(spot or 100.0)
        rate = float(rate or 0.05)
        div_yield = float(div_yield or 0.01)
        noise = float(noise or 0.003)
        max_iter = int(max_iter or 300)

        # Generate synthetic market data
        market = generate_synthetic_market_surface(S=spot, r=rate, q=div_yield)
        market_iv = market["iv_grid"]

        # Add noise
        if noise > 0:
            rng = np.random.RandomState(42)
            market_iv = market_iv + rng.normal(0, noise, market_iv.shape)
            market_iv = np.clip(market_iv, 0.02, 2.0)

        # Build arb-free surface
        builder = ArbitrageFreeSVIBuilder(max_iter=max_iter)
        result = builder.build_surface(
            spot=spot,
            strikes=market["strikes"],
            expiries=market["expiries"],
            market_iv_grid=market_iv,
            r=rate,
            q=div_yield,
        )

        adj_iv = result["adj_iv_grid"]
        K_grid = result["K_grid"] / spot  # Normalize to moneyness
        T_grid = result["T_grid"]

        # 3D surface
        fig_3d = go.Figure(data=[
            go.Surface(
                x=K_grid,
                y=T_grid,
                z=adj_iv * 100,
                colorscale="Inferno",
                name="Arb-Free SVI",
                colorbar=dict(
                    title=dict(text="IV (%)", font=dict(color=TEXT_SECONDARY, size=11)),
                    tickfont=dict(color=TEXT_SECONDARY, size=10),
                    len=0.6,
                ),
                hovertemplate=(
                    "Moneyness: %{x:.2f}<br>"
                    "Expiry: %{y:.2f}Y<br>"
                    "IV: %{z:.1f}%<extra></extra>"
                ),
            )
        ])
        fig_3d.update_layout(
            title="Arbitrage-Free SVI Surface",
            height=520,
            scene=dict(
                xaxis=dict(title="Moneyness (K/S)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
                yaxis=dict(title="Expiry (Years)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
                zaxis=dict(title="IV (%)", backgroundcolor=BG_SECONDARY, gridcolor=BORDER_LIGHT, color=TEXT_SECONDARY),
                bgcolor=BG_PRIMARY,
                camera=SURFACE_CAMERA,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        # 2D smiles at select expiries
        fig_2d = go.Figure()
        expiry_arr = result["expiries"]
        target_tenors = [0.083, 0.25, 0.5, 1.0, 2.0]
        tenor_labels = ["1M", "3M", "6M", "1Y", "2Y"]
        moneyness_arr = result["strikes"] / spot

        for target_T, label, color in zip(target_tenors, tenor_labels, SMILE_COLORS):
            idx = int(np.argmin(np.abs(expiry_arr - target_T)))
            if idx < len(adj_iv):
                fig_2d.add_trace(go.Scatter(
                    x=moneyness_arr,
                    y=adj_iv[idx] * 100,
                    mode="lines",
                    name=f"{label} (T={expiry_arr[idx]:.2f})",
                    line=dict(color=color, width=2),
                ))

        # Overlay raw market data for comparison
        raw_iv = result.get("raw_iv_grid")
        if raw_iv is not None and len(expiry_arr) > 0:
            mid_idx = len(expiry_arr) // 2
            fig_2d.add_trace(go.Scatter(
                x=moneyness_arr,
                y=raw_iv[mid_idx] * 100,
                mode="markers",
                name="Raw (mid-expiry)",
                marker=dict(color=TEXT_TERTIARY, size=4, symbol="x"),
            ))

        fig_2d.update_layout(
            title="Arb-Free Smile Curves vs Raw Fit",
            height=320,
            xaxis=dict(title="Moneyness (K/S)"),
            yaxis=dict(title="IV (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig_3d, fig_2d

    except (ValueError, KeyError, TypeError, IndexError) as e:
        err_msg = f"Arb-Free SVI error: {str(e)[:120]}"
        return empty_figure(err_msg), empty_figure(err_msg)
