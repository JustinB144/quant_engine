# Dash UI Quick Start Guide

## Installation

```bash
pip install dash dash-bootstrap-components plotly pandas numpy flask-caching
```

## Start Development Server

```bash
cd /sessions/sleepy-nice-ritchie/mnt/quant_engine
python -m dash_ui.server
```

Server runs at: **http://localhost:8050**

## File Structure

```
dash_ui/
├── __init__.py           # Package (v1.0.0)
├── theme.py              # Colors, fonts, Plotly template
├── app.py                # Dash app factory
├── server.py             # Entry point + caching
│
├── data/
│   ├── cache.py          # Flask-Caching wrapper
│   └── loaders.py        # Data loading functions
│
├── components/
│   ├── metric_card.py    # KPI cards
│   ├── trade_table.py    # DataTable
│   ├── chart_utils.py    # Plotly functions
│   ├── regime_badge.py   # Regime indicator
│   ├── alert_banner.py   # Alerts
│   └── sidebar.py        # Navigation
│
└── pages/
    ├── dashboard.py      # Main page (/)
    ├── system_health.py  # Health checks (/health)
    └── ...               # 9 pages total
```

## Quick Code Snippets

### Create a Metric Card
```python
from dash_ui.components import metric_card

card = metric_card(
    title="Sharpe Ratio",
    value="1.47",
    delta="+0.12",
    fmt="number"
)
```

### Create a Chart
```python
from dash_ui.components import chart_utils
import pandas as pd

fig = chart_utils.line_chart(
    data_dict={
        "Portfolio": (dates, portfolio_returns),
        "SPY": (dates, spy_returns),
    },
    title="Cumulative Returns",
    yaxis_title="Return"
)
```

### Load Data with Caching
```python
from dash_ui.data.cache import cached
from dash_ui.data.loaders import load_trades, compute_risk_metrics

@cached(timeout=300)  # Cache 5 minutes
def get_trades_and_metrics():
    trades = load_trades(path)
    metrics = compute_risk_metrics(returns)
    return trades, metrics
```

### Create Alert
```python
from dash_ui.components import alert_banner

banner = alert_banner(
    "Model retrain triggered",
    severity="warning"
)
```

### Create Trade Table
```python
from dash_ui.components import trade_table

table = trade_table(
    id="trades-table",
    columns=[
        {"name": "Entry", "id": "entry_date"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Return %", "id": "net_return"},
    ],
    data=trades.to_dict("records")
)
```

### Access Theme Colors
```python
from dash_ui.theme import (
    BG_PRIMARY,      # "#0d1117"
    ACCENT_GREEN,    # "#3fb950"
    ACCENT_RED,      # "#f85149"
    ACCENT_BLUE,     # "#58a6ff"
    REGIME_COLORS,   # {0: green, 1: red, ...}
    format_pct,      # 0.123 → "12.3%"
    format_number,   # 1234.5 → "1,234.50"
)
```

## Available Chart Types

```python
from dash_ui.components.chart_utils import (
    line_chart,           # Multi-line
    area_chart,           # Filled area
    bar_chart,            # Bars (horizontal/vertical)
    heatmap_chart,        # Annotated heatmap
    surface_3d,           # 3D surface
    equity_curve,         # With drawdown shading
    regime_timeline,      # Colored regime bands
    dual_axis_chart,      # Two y-axes
    candlestick_chart,    # OHLCV + volume
    scatter_chart,        # With color/size encoding
    radar_chart,          # Polar
    histogram_chart,      # Distribution
)
```

## Available Data Loaders

```python
from dash_ui.data.loaders import (
    load_trades,                  # CSV → DataFrame
    build_portfolio_returns,      # Trades → Series
    load_benchmark_returns,       # SPY from cache
    load_factor_proxies,          # Tech/momentum factors
    compute_risk_metrics,         # Sharpe, sortino, DD, etc.
    compute_regime_payload,       # HMM regime detection
    compute_model_health,         # CV gap, IC drift, etc.
    load_feature_importance,      # Global + regime importance
    collect_health_data,          # Full system audit
)
```

## System Health Assessment

```python
from dash_ui.data.loaders import collect_health_data

health = collect_health_data()
# Returns SystemHealthPayload with:
#   - overall_score (0-100)
#   - overall_status ("PASS" / "WARN" / "FAIL")
#   - 5 audit dimensions:
#     1. Data integrity & survivorship
#     2. Promotion contract gates
#     3. Walk-forward validation
#     4. Execution cost model
#     5. Feature/knob complexity
```

## Create a New Page

1. Create `dash_ui/pages/my_page.py`:
```python
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/my-page")

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("My Page"),
            ], md=12)
        ]),
    ], fluid=True)
])
```

2. Page automatically routed to `/my-page`
3. Use components from `dash_ui.components`

## Production Deployment

```bash
# With Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:8050 \
  --timeout 120 \
  --log-level info \
  quant_engine.dash_ui.server:server
```

## Performance Tips

1. Use `@cached()` decorator on expensive functions:
```python
@cached(timeout=600)
def load_expensive_data():
    return compute_something()
```

2. Use `virtualization="windowed"` in large tables
3. Keep charts to <500 points for smooth interaction
4. Pagination: 50 rows per page (default)

## Color Reference

```python
# Backgrounds
BG_PRIMARY = "#0d1117"      # Main
BG_SECONDARY = "#161b22"    # Panels
BG_TERTIARY = "#1c2028"     # Elements
BG_SIDEBAR = "#010409"      # Sidebar

# Text
TEXT_PRIMARY = "#ffffff"    # Main
TEXT_SECONDARY = "#c9d1d9"  # Secondary
TEXT_TERTIARY = "#8b949e"   # Muted

# Accents
ACCENT_GREEN = "#3fb950"    # Success/bull
ACCENT_RED = "#f85149"      # Danger/bear
ACCENT_BLUE = "#58a6ff"     # Primary
ACCENT_AMBER = "#d29922"    # Warning/vol
ACCENT_PURPLE = "#bc8cff"   # Tertiary
ACCENT_CYAN = "#39d2c0"     # Quaternary
ACCENT_ORANGE = "#f0883e"   # Additional

# Regime
REGIME_COLORS = {
    0: "#3fb950",  # Trending Bull
    1: "#f85149",  # Trending Bear
    2: "#58a6ff",  # Mean Reverting
    3: "#d29922",  # High Volatility
}
```

## Troubleshooting

**Issue: "Port 8050 already in use"**
```bash
# Kill existing process
lsof -ti:8050 | xargs kill -9
```

**Issue: Cache not working**
- Ensure `/tmp/qe_dash_cache` is writable
- Check Flask app is initialized with `init_cache(app)`

**Issue: Plotly charts not showing**
- Import `create_figure()` from theme
- All figures must use the template

## API Reference

### theme.py
- `create_figure(**kwargs) → go.Figure`
- `format_pct(value, decimals=1) → str`
- `format_number(value, decimals=2) → str`
- `metric_color(value, positive_is_good=True) → str`
- `apply_plotly_template()`

### cache.py
- `init_cache(app) → None`
- `@cached(timeout=60)` decorator

### components
- `metric_card(title, value, delta, fmt, color) → dbc.Card`
- `trade_table(id, columns, data) → html.Div`
- `regime_badge(regime_code) → html.Div`
- `alert_banner(message, severity, icon, dismissable) → dbc.Alert`
- `chart_utils.*` (12 chart functions)

## Resources

- Dash docs: https://dash.plotly.com
- Plotly docs: https://plotly.com/python
- Bootstrap docs: https://dash-bootstrap-components.opensource.faculty.ai
- Quant Engine config: `config.py`
