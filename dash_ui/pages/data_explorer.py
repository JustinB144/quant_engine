"""
Data Explorer -- OHLCV visualization and quality analysis.

Provides interactive data loading, candlestick/price charting with SMA overlays,
volume analysis, and data quality reporting for any ticker in the universe.
"""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate

from quant_engine.dash_ui.theme import (
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BORDER,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_TERTIARY,
    CHART_COLORS, empty_figure, enhance_time_series,
)
from quant_engine.dash_ui.components.page_header import create_page_header

dash.register_page(__name__, path="/data-explorer", name="Data Explorer", order=2)

# ---------------------------------------------------------------------------
# Universe imports
# ---------------------------------------------------------------------------
try:
    from quant_engine.config import UNIVERSE_QUICK, UNIVERSE_FULL
except ImportError:
    UNIVERSE_QUICK = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA",
        "JPM", "UNH", "HD", "V", "DDOG", "CRWD", "CAVA",
    ]
    UNIVERSE_FULL = UNIVERSE_QUICK + [
        "AMD", "INTC", "CRM", "ADBE", "ORCL", "JNJ", "PFE", "ABBV",
        "MRK", "LLY", "TMO", "ABT", "HD", "NKE", "SBUX", "MCD",
        "TGT", "COST", "BAC", "GS", "MS", "BLK", "MA",
        "CAT", "DE", "GE", "HON", "BA", "LMT",
    ]

# ---------------------------------------------------------------------------
# Shared style helpers
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
# Client-side data store (serialized as JSON in dcc.Store)
# ---------------------------------------------------------------------------
# We keep all loaded ticker data in a dict-of-dicts keyed by ticker symbol.

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div([
    # Hidden stores
    dcc.Store(id="de-loaded-data", data={}),
    dcc.Store(id="de-selected-ticker", data=None),

    # Page title
    create_page_header("Data Explorer", subtitle="Market data analysis and quality checks"),

    # ── Controls row ──────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Universe", style=_LABEL_STYLE),
            dcc.Dropdown(
                id="de-universe-dropdown",
                options=[
                    {"label": "QUICK (14 tickers)", "value": "QUICK"},
                    {"label": "FULL (50+ tickers)", "value": "FULL"},
                    {"label": "CUSTOM", "value": "CUSTOM"},
                ],
                value="QUICK",
                clearable=False,
                style={"fontSize": "12px"},
            ),
        ], width=2),
        dbc.Col([
            html.Label("Timeframe", style=_LABEL_STYLE),
            dcc.Dropdown(
                id="de-timeframe-dropdown",
                options=[
                    {"label": "Daily", "value": "daily"},
                    {"label": "4 Hour", "value": "4h"},
                    {"label": "1 Hour", "value": "1h"},
                    {"label": "30 Min", "value": "30m"},
                ],
                value="daily",
                clearable=False,
                style={"fontSize": "12px"},
            ),
        ], width=2),
        dbc.Col([
            html.Label("Ticker(s)", style=_LABEL_STYLE),
            dbc.Input(
                id="de-ticker-input",
                placeholder="e.g. AAPL, MSFT",
                type="text",
                size="sm",
                style={"backgroundColor": BG_PRIMARY, "borderColor": BORDER,
                       "color": TEXT_PRIMARY, "fontSize": "12px"},
            ),
        ], width=2),
        dbc.Col([
            html.Label("\u00a0", style=_LABEL_STYLE),
            html.Div([
                dbc.Button(
                    "Load Data",
                    id="de-load-btn",
                    color="primary",
                    size="sm",
                    className="me-2",
                ),
                dbc.Button(
                    "View Quality",
                    id="de-quality-btn",
                    color="secondary",
                    size="sm",
                    outline=True,
                ),
            ]),
        ], width=3),
        dbc.Col([
            html.Label("\u00a0", style=_LABEL_STYLE),
            html.Div(
                id="de-status-text",
                children="Ready. Select universe and click Load Data.",
                style={"fontSize": "11px", "color": TEXT_TERTIARY,
                       "fontFamily": "Menlo, monospace", "paddingTop": "6px"},
            ),
        ], width=4),
    ], className="mb-3"),

    # ── Main panels ───────────────────────────────────────────────────
    dbc.Row([
        # Left: ticker list
        dbc.Col([
            html.Div([
                html.Div("Loaded Tickers", style={
                    "fontSize": "13px", "fontWeight": "600", "color": TEXT_PRIMARY,
                    "marginBottom": "12px", "paddingBottom": "8px",
                    "borderBottom": f"1px solid {BORDER}",
                }),
                html.Div(
                    id="de-ticker-list-container",
                    children=html.Div(
                        "No data loaded",
                        style={"fontSize": "11px", "color": TEXT_TERTIARY, "padding": "8px"},
                    ),
                    style={"maxHeight": "600px", "overflowY": "auto"},
                ),
            ], style=_CARD_STYLE),
        ], width=3),

        # Right: charts
        dbc.Col([
            # Price chart
            html.Div([
                html.Div("Price Chart", style={
                    "fontSize": "13px", "fontWeight": "600", "color": TEXT_PRIMARY,
                    "marginBottom": "8px",
                }),
                dcc.Loading(
                    dcc.Graph(
                        id="de-price-chart",
                        figure=empty_figure("Select a ticker to view price data"),
                        config={"displayModeBar": True, "scrollZoom": True},
                        style={"height": "400px"},
                    ),
                    type="dot",
                    color=ACCENT_BLUE,
                ),
            ], style=_CARD_STYLE),

            # Volume chart
            html.Div([
                html.Div("Volume", style={
                    "fontSize": "13px", "fontWeight": "600", "color": TEXT_PRIMARY,
                    "marginBottom": "8px",
                }),
                dcc.Loading(
                    dcc.Graph(
                        id="de-volume-chart",
                        figure=empty_figure("Select a ticker to view volume data"),
                        config={"displayModeBar": True, "scrollZoom": True},
                        style={"height": "200px"},
                    ),
                    type="dot",
                    color=ACCENT_BLUE,
                ),
            ], style=_CARD_STYLE),
        ], width=9),
    ]),

    # ── Bottom stats bar ──────────────────────────────────────────────
    html.Div(
        id="de-stats-bar",
        style={
            "backgroundColor": BG_TERTIARY,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "display": "flex",
            "gap": "32px",
            "flexWrap": "wrap",
        },
    ),

    # ── Quality modal ─────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle("Data Quality Report"),
            close_button=True,
            style={"backgroundColor": BG_SECONDARY, "borderBottom": f"1px solid {BORDER}"},
        ),
        dbc.ModalBody(
            id="de-quality-body",
            style={"backgroundColor": BG_PRIMARY, "color": TEXT_PRIMARY},
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="de-quality-close-btn", color="secondary", size="sm"),
            style={"backgroundColor": BG_SECONDARY, "borderTop": f"1px solid {BORDER}"},
        ),
    ], id="de-quality-modal", size="lg", is_open=False,
       style={"color": TEXT_PRIMARY}),
])


# ---------------------------------------------------------------------------
# Helper: download or generate data
# ---------------------------------------------------------------------------
def _download_ticker(ticker: str, timeframe: str = "daily", period: str = "2y") -> pd.DataFrame:
    """Load OHLCV data for a ticker and timeframe.

    For intraday timeframes, loads from local IBKR cache only.
    For daily, tries local cache first then falls back to demo data.
    """
    if timeframe != "daily":
        # Intraday: local cache only
        try:
            from quant_engine.data.local_cache import load_intraday_ohlcv
            df = load_intraday_ohlcv(ticker, timeframe)
            if df is not None and len(df) > 0:
                return df
        except (ImportError, OSError, ValueError, KeyError, TypeError):
            pass
        return None

    # Daily: try local cache first
    try:
        from quant_engine.data.local_cache import load_ohlcv_with_meta
        df, _meta, _path = load_ohlcv_with_meta(ticker)
        if df is not None and len(df) > 10:
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            return df
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        pass

    # Fallback: generate realistic demo data
    return _generate_demo_data(ticker)


def _generate_demo_data(ticker: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data seeded by ticker name."""
    seed = sum(ord(c) for c in ticker) + 42
    rng = np.random.RandomState(seed)
    n_days = 504  # ~2 years of trading days
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n_days)

    base_price = 50.0 + rng.uniform(0, 400)
    daily_returns = rng.normal(0.0003, 0.018, n_days)
    prices = base_price * np.exp(np.cumsum(daily_returns))

    daily_vol = np.abs(rng.normal(0, 0.012, n_days))
    high = prices * (1.0 + daily_vol)
    low = prices * (1.0 - daily_vol)
    open_prices = low + rng.uniform(0.3, 0.7, n_days) * (high - low)
    volume = (rng.lognormal(mean=15, sigma=0.8, size=n_days)).astype(int)

    df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)
    return df


def _compute_sma(close: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with NaN fill for insufficient data."""
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) >= window:
        cumsum = np.cumsum(close)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        out[window - 1:] = cumsum[window - 1:] / window
    return out


def _df_to_store(df: pd.DataFrame) -> dict:
    """Serialize a DataFrame to a JSON-safe dict for dcc.Store."""
    return {
        "index": [d.isoformat() for d in df.index],
        "Open": df["Open"].tolist(),
        "High": df["High"].tolist(),
        "Low": df["Low"].tolist(),
        "Close": df["Close"].tolist(),
        "Volume": df["Volume"].tolist(),
    }


def _store_to_df(data: dict) -> pd.DataFrame:
    """Deserialize a dict from dcc.Store back to a DataFrame."""
    df = pd.DataFrame({
        "Open": data["Open"],
        "High": data["High"],
        "Low": data["Low"],
        "Close": data["Close"],
        "Volume": data["Volume"],
    }, index=pd.to_datetime(data["index"]))
    return df


# ---------------------------------------------------------------------------
# Callback: Load data
# ---------------------------------------------------------------------------
@callback(
    Output("de-loaded-data", "data"),
    Output("de-status-text", "children"),
    Input("de-load-btn", "n_clicks"),
    State("de-universe-dropdown", "value"),
    State("de-timeframe-dropdown", "value"),
    State("de-ticker-input", "value"),
    State("de-loaded-data", "data"),
    prevent_initial_call=True,
)
def load_data(n_clicks, universe, timeframe, custom_tickers, existing_data):
    """Load selected ticker OHLCV data into the store and return a status summary."""
    if not n_clicks:
        raise PreventUpdate

    timeframe = timeframe or "daily"

    if universe == "QUICK":
        tickers = list(UNIVERSE_QUICK)
    elif universe == "FULL":
        tickers = list(UNIVERSE_FULL)
    elif universe == "CUSTOM" and custom_tickers:
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    elif custom_tickers:
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    else:
        return existing_data or {}, "No tickers specified. Enter tickers or select a universe."

    if not tickers:
        return existing_data or {}, "No valid tickers found."

    store = dict(existing_data) if existing_data else {}
    loaded = []
    failed = []

    for ticker in tickers:
        try:
            df = _download_ticker(ticker, timeframe=timeframe)
            if df is not None and len(df) > 0:
                store[ticker] = _df_to_store(df)
                loaded.append(ticker)
            else:
                failed.append(ticker)
        except (ValueError, KeyError, TypeError, IndexError):
            failed.append(ticker)

    tf_label = timeframe if timeframe == "daily" else timeframe.upper()
    msg_parts = []
    if loaded:
        msg_parts.append(f"Loaded {len(loaded)} ticker(s) [{tf_label}]")
    if failed:
        msg_parts.append(f"Failed: {', '.join(failed)}")
    msg_parts.append(f"Total in cache: {len(store)}")
    status = " | ".join(msg_parts)

    return store, status


# ---------------------------------------------------------------------------
# Callback: Render ticker list
# ---------------------------------------------------------------------------
@callback(
    Output("de-ticker-list-container", "children"),
    Input("de-loaded-data", "data"),
    Input("de-selected-ticker", "data"),
)
def render_ticker_list(loaded_data, selected_ticker):
    """Render the selectable ticker list with active-state highlighting and bar counts."""
    if not loaded_data:
        return html.Div(
            "No data loaded",
            style={"fontSize": "11px", "color": TEXT_TERTIARY, "padding": "8px"},
        )

    items = []
    for ticker in sorted(loaded_data.keys()):
        data = loaded_data[ticker]
        n_bars = len(data.get("Close", []))
        is_active = ticker == selected_ticker
        items.append(
            dbc.ListGroupItem(
                html.Div([
                    html.Span(ticker, style={
                        "fontWeight": "600",
                        "fontFamily": "Menlo, monospace",
                        "fontSize": "12px",
                        "color": ACCENT_BLUE if is_active else TEXT_PRIMARY,
                    }),
                    html.Span(f"  {n_bars} bars", style={
                        "fontSize": "10px", "color": TEXT_TERTIARY,
                        "fontFamily": "Menlo, monospace",
                    }),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center"}),
                id={"type": "de-ticker-item", "ticker": ticker},
                action=True,
                active=is_active,
                style={
                    "backgroundColor": BG_TERTIARY if is_active else BG_SECONDARY,
                    "borderColor": ACCENT_BLUE if is_active else BORDER,
                    "padding": "8px 12px",
                    "cursor": "pointer",
                    "fontSize": "12px",
                },
                n_clicks=0,
            )
        )

    return dbc.ListGroup(items, flush=True)


# ---------------------------------------------------------------------------
# Callback: Ticker selection via pattern-matching click
# ---------------------------------------------------------------------------
@callback(
    Output("de-selected-ticker", "data"),
    Input({"type": "de-ticker-item", "ticker": dash.ALL}, "n_clicks"),
    State("de-loaded-data", "data"),
    prevent_initial_call=True,
)
def select_ticker(n_clicks_list, loaded_data):
    """Persist the ticker symbol clicked in the pattern-matched ticker list."""
    if not loaded_data or not any(n_clicks_list):
        raise PreventUpdate

    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict) and "ticker" in triggered:
        return triggered["ticker"]
    raise PreventUpdate


# ---------------------------------------------------------------------------
# Callback: Update price chart
# ---------------------------------------------------------------------------
@callback(
    Output("de-price-chart", "figure"),
    Input("de-selected-ticker", "data"),
    State("de-loaded-data", "data"),
)
def update_price_chart(ticker, loaded_data):
    """Build the main price chart (candles + SMA overlays) for the selected ticker."""
    if not ticker or not loaded_data or ticker not in loaded_data:
        return empty_figure("Select a ticker to view price data")

    df = _store_to_df(loaded_data[ticker])
    close = df["Close"].values.astype(float)
    sma20 = _compute_sma(close, 20)
    sma50 = _compute_sma(close, 50)

    fig = go.Figure()

    # Try candlestick first
    try:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=ACCENT_GREEN,
            decreasing_line_color=ACCENT_RED,
            increasing_fillcolor=ACCENT_GREEN,
            decreasing_fillcolor=ACCENT_RED,
        ))
    except (ValueError, KeyError, TypeError, IndexError):
        # Fallback: scatter with high/low fill_between
        fig.add_trace(go.Scatter(
            x=df.index, y=df["High"].values,
            mode="lines", line=dict(width=0),
            showlegend=False, name="High",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Low"].values,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(88,166,255,0.1)",
            showlegend=False, name="Low",
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=close,
            mode="lines", line=dict(color=ACCENT_BLUE, width=1.5),
            name="Close",
        ))

    # SMA overlays
    fig.add_trace(go.Scatter(
        x=df.index, y=sma20,
        mode="lines", line=dict(color=ACCENT_AMBER, width=1, dash="dot"),
        name="SMA 20",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=sma50,
        mode="lines", line=dict(color=ACCENT_CYAN, width=1, dash="dash"),
        name="SMA 50",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} -- Price", font=dict(size=13)),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(title="Price ($)"),
        height=400,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=10),
        ),
    )
    enhance_time_series(fig)
    return fig


# ---------------------------------------------------------------------------
# Callback: Update volume chart
# ---------------------------------------------------------------------------
@callback(
    Output("de-volume-chart", "figure"),
    Input("de-selected-ticker", "data"),
    State("de-loaded-data", "data"),
)
def update_volume_chart(ticker, loaded_data):
    """Build the color-coded volume bar chart for the selected ticker."""
    if not ticker or not loaded_data or ticker not in loaded_data:
        return empty_figure("Select a ticker to view volume data")

    df = _store_to_df(loaded_data[ticker])
    close = df["Close"].values.astype(float)

    # Determine bar colors: green if close >= previous close, red otherwise
    direction = np.zeros(len(close))
    direction[1:] = np.where(close[1:] >= close[:-1], 1.0, -1.0)
    direction[0] = 1.0
    colors = [ACCENT_GREEN if d >= 0 else ACCENT_RED for d in direction]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"].values,
        marker_color=colors,
        name="Volume",
        opacity=0.7,
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} -- Volume", font=dict(size=13)),
        yaxis=dict(title="Volume"),
        xaxis=dict(type="date"),
        height=200,
        margin=dict(l=50, r=20, t=35, b=30),
        showlegend=False,
        bargap=0.1,
    )
    return fig


# ---------------------------------------------------------------------------
# Callback: Update stats bar
# ---------------------------------------------------------------------------
@callback(
    Output("de-stats-bar", "children"),
    Input("de-selected-ticker", "data"),
    State("de-loaded-data", "data"),
)
def update_stats_bar(ticker, loaded_data):
    """Summarize key ticker statistics shown in the bottom stats strip."""
    if not ticker or not loaded_data or ticker not in loaded_data:
        return html.Div(
            "Select a ticker to view statistics",
            style={"fontSize": "11px", "color": TEXT_TERTIARY},
        )

    df = _store_to_df(loaded_data[ticker])
    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    last_price = close[-1] if len(close) > 0 else 0.0
    daily_change = ((close[-1] / close[-2]) - 1.0) * 100 if len(close) > 1 else 0.0
    high_52w = float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
    low_52w = float(np.min(close[-252:])) if len(close) >= 252 else float(np.min(close))
    avg_volume = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
    total_bars = len(close)
    date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"

    change_color = ACCENT_GREEN if daily_change >= 0 else ACCENT_RED

    def _stat(label, value, color=TEXT_PRIMARY):
        return html.Div([
            html.Div(label, style={"fontSize": "10px", "color": TEXT_TERTIARY,
                                   "fontFamily": "Menlo, monospace"}),
            html.Div(value, style={"fontSize": "13px", "fontWeight": "600", "color": color,
                                   "fontFamily": "Menlo, monospace"}),
        ])

    return [
        _stat("Ticker", ticker, ACCENT_BLUE),
        _stat("Last Price", f"${last_price:,.2f}"),
        _stat("Day Change", f"{daily_change:+.2f}%", change_color),
        _stat("52W High", f"${high_52w:,.2f}"),
        _stat("52W Low", f"${low_52w:,.2f}"),
        _stat("Avg Vol (20d)", f"{avg_volume:,.0f}"),
        _stat("Total Bars", f"{total_bars:,}"),
        _stat("Range", date_range, TEXT_SECONDARY),
    ]


# ---------------------------------------------------------------------------
# Callback: Quality modal toggle
# ---------------------------------------------------------------------------
@callback(
    Output("de-quality-modal", "is_open"),
    Output("de-quality-body", "children"),
    Input("de-quality-btn", "n_clicks"),
    Input("de-quality-close-btn", "n_clicks"),
    State("de-quality-modal", "is_open"),
    State("de-loaded-data", "data"),
    prevent_initial_call=True,
)
def toggle_quality_modal(open_clicks, close_clicks, is_open, loaded_data):
    """Open/close the data-quality modal and populate the per-ticker quality table."""
    trigger = ctx.triggered_id
    if trigger == "de-quality-close-btn":
        return False, no_update

    if trigger == "de-quality-btn":
        if not loaded_data:
            return True, html.Div(
                "No data loaded. Load data first to generate a quality report.",
                style={"color": TEXT_TERTIARY, "fontSize": "12px"},
            )

        rows = []
        for ticker in sorted(loaded_data.keys()):
            df = _store_to_df(loaded_data[ticker])
            n_bars = len(df)
            close = df["Close"].values.astype(float)
            volume = df["Volume"].values.astype(float)

            # Missing values
            missing_count = int(df.isnull().sum().sum())

            # Zero volume days
            zero_vol = int(np.sum(volume == 0))
            zero_vol_pct = (zero_vol / n_bars * 100) if n_bars > 0 else 0.0

            # Extreme returns (|return| > 20%)
            if len(close) > 1:
                returns = np.diff(close) / close[:-1]
                extreme_count = int(np.sum(np.abs(returns) > 0.20))
                max_return = float(np.max(returns)) * 100
                min_return = float(np.min(returns)) * 100
            else:
                extreme_count = 0
                max_return = 0.0
                min_return = 0.0

            # Quality score
            issues = 0
            if missing_count > 0:
                issues += 1
            if zero_vol_pct > 25:
                issues += 2
            elif zero_vol_pct > 5:
                issues += 1
            if extreme_count > 5:
                issues += 2
            elif extreme_count > 0:
                issues += 1

            if issues == 0:
                quality = "GOOD"
                q_color = ACCENT_GREEN
            elif issues <= 2:
                quality = "FAIR"
                q_color = ACCENT_AMBER
            else:
                quality = "POOR"
                q_color = ACCENT_RED

            rows.append(html.Tr([
                html.Td(ticker, style={"fontWeight": "600", "color": ACCENT_BLUE}),
                html.Td(f"{n_bars:,}"),
                html.Td(str(missing_count),
                         style={"color": ACCENT_RED if missing_count > 0 else TEXT_SECONDARY}),
                html.Td(f"{zero_vol} ({zero_vol_pct:.1f}%)",
                         style={"color": ACCENT_AMBER if zero_vol_pct > 5 else TEXT_SECONDARY}),
                html.Td(str(extreme_count),
                         style={"color": ACCENT_RED if extreme_count > 0 else TEXT_SECONDARY}),
                html.Td(f"{max_return:+.1f}% / {min_return:+.1f}%"),
                html.Td(quality, style={"color": q_color, "fontWeight": "600"}),
            ]))

        table_header = html.Thead(html.Tr([
            html.Th("Ticker"),
            html.Th("Bars"),
            html.Th("Missing"),
            html.Th("Zero Vol"),
            html.Th("Extreme Ret"),
            html.Th("Max / Min Ret"),
            html.Th("Quality"),
        ]))

        cell_style = {
            "fontSize": "11px", "fontFamily": "Menlo, monospace", "padding": "6px 10px",
            "borderColor": BORDER, "color": TEXT_SECONDARY,
        }
        header_style = {
            "fontSize": "11px", "fontFamily": "Menlo, monospace", "padding": "8px 10px",
            "borderColor": BORDER, "color": TEXT_PRIMARY, "fontWeight": "600",
            "backgroundColor": BG_TERTIARY,
        }

        # Apply explicit dark theme styles — dbc.Table dark=True conflicts
        # with our custom palette by setting near-black text on dark bg.
        for row in rows:
            for cell in row.children:
                if isinstance(cell, html.Td) and (not cell.style or "color" not in cell.style):
                    existing = dict(cell.style) if cell.style else {}
                    existing.setdefault("color", TEXT_SECONDARY)
                    existing.setdefault("borderColor", BORDER)
                    cell.style = existing

        for cell in table_header.children.children:
            if isinstance(cell, html.Th):
                existing = dict(cell.style) if cell.style else {}
                existing.update({"color": TEXT_PRIMARY, "backgroundColor": BG_TERTIARY})
                cell.style = existing

        table = dbc.Table(
            [table_header, html.Tbody(rows)],
            bordered=True, hover=True, size="sm",
            style={"backgroundColor": BG_SECONDARY, "color": TEXT_SECONDARY,
                   "marginBottom": "0"},
        )

        # Apply styles via wrapping -- Dash tables auto-use CSS class overrides
        summary_text = (
            f"Analyzed {len(loaded_data)} ticker(s). "
            f"Bars range from "
            f"{min(len(loaded_data[t].get('Close', [])) for t in loaded_data):,} to "
            f"{max(len(loaded_data[t].get('Close', [])) for t in loaded_data):,}."
        )

        return True, html.Div([
            html.Div(summary_text, style={
                "fontSize": "12px", "color": TEXT_SECONDARY, "marginBottom": "12px",
                "fontFamily": "Menlo, monospace",
            }),
            table,
            html.Div([
                html.Hr(style={"borderColor": BORDER, "margin": "16px 0 12px"}),
                html.Div("Quality Criteria:", style={
                    "fontSize": "11px", "fontWeight": "600", "color": TEXT_PRIMARY,
                    "marginBottom": "8px",
                }),
                html.Ul([
                    html.Li("Missing Values: NaN/null entries across all OHLCV columns"),
                    html.Li("Zero Volume: Trading days with zero reported volume"),
                    html.Li("Extreme Returns: Daily returns exceeding +/-20%"),
                    html.Li("GOOD = no issues, FAIR = minor issues, POOR = significant issues"),
                ], style={"fontSize": "10px", "color": TEXT_TERTIARY,
                          "fontFamily": "Menlo, monospace", "paddingLeft": "20px"}),
            ]),
        ])

    return is_open, no_update
