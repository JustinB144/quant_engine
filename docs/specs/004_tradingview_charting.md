# Feature Spec: TradingView Advanced Charting with Custom Data Feed

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~14 hours across 7 tasks

---

## Why

The current Data Explorer page uses `lightweight-charts` for a basic candlestick chart with only SMA 20/50 overlays. The user has a TradingView subscription, a library of custom indicators, and intraday data (5min, 15min, 30min, 1hr parquets already cached). None of this is accessible through the UI. The charting should support multiple timeframes, technical indicators from the existing `indicators/indicators.py` (90+ indicators), and the professional TradingView interface that the user already knows.

## What

Replace the basic `lightweight-charts` candlestick with the TradingView Advanced Chart widget (free embed) for quick access to TradingView's built-in indicators, AND build a custom UDF (Universal Data Feed) backend endpoint that serves the user's cached OHLCV data (daily + intraday) to the TradingView widget. Done means: the Data Explorer page shows a full TradingView chart with timeframe selector, indicator overlay support, and data loaded from the local cache.

## Constraints

### Must-haves
- TradingView Advanced Chart widget embedded via iframe (free, no Charting Library license needed)
- Symbols resolve to local ticker names (AAPL, MSFT, etc.)
- Timeframes supported: 5min, 15min, 30min, 1hr, 1day (matching cached parquets)
- Volume displayed
- User can add TradingView's built-in indicators (RSI, MACD, Bollinger, etc.)
- Dark theme matching Bloomberg-dark UI
- Fallback: if TradingView widget fails to load (network issues), show the existing lightweight-charts candlestick

### Must-nots
- Do NOT require TradingView Charting Library license (use free Advanced Chart widget OR existing lightweight-charts with enhancements)
- Do NOT remove existing charting code — keep as fallback
- Do NOT store any user credentials or API keys for TradingView

### Out of scope
- Custom Pine Script indicators
- Real-time streaming data (use cached historical only)
- Drawing tools persistence
- Multi-chart layouts

### Important note on TradingView widget limitations
The free Advanced Chart widget uses TradingView's own data feed — it cannot serve custom/local data. Two approaches:

**Option A (Recommended): Enhanced lightweight-charts with indicator overlays**
- Keep `lightweight-charts` as primary
- Add timeframe selector (5min, 15min, 30min, 1hr, 1day)
- Add indicator overlay panel (compute indicators server-side via existing `indicators.py`)
- Add volume bars, drawing tools
- Result: Full-featured chart with YOUR data and YOUR indicators

**Option B: TradingView widget for reference + lightweight-charts for custom data**
- Embed TradingView widget as a "Market Reference" panel (uses TV data)
- Keep lightweight-charts as "Custom Data" panel (uses your cached data)
- Side-by-side comparison

This spec implements **Option A** since it uses your actual cached data and indicators.

## Current State

### Key files
| File | Role | Notes |
|------|------|-------|
| `frontend/src/components/charts/CandlestickChart.tsx` | Lightweight-charts candlestick with SMA 20/50 | Only supports daily timeframe, hardcoded SMA periods |
| `frontend/src/pages/DataExplorerPage.tsx` | Data explorer with universe selector + chart | Fetches `/data/ticker/{ticker}?years=2` — only daily |
| `indicators/indicators.py` | 90+ technical indicators | Fully implemented but not exposed via API |
| `data/cache/` | Cached parquets: `{TICKER}_{timeframe}_{start}_{end}.parquet` | Has 5min, 15min, 30min, 1hr, daily for many tickers |
| `api/routers/data_explorer.py` | `/data/ticker/{ticker}` endpoint | Only returns daily bars |

### Existing charting patterns
- `lightweight-charts@4.2.0` already installed
- Charts use `ChartContainer` wrapper for loading/error/empty states
- Bloomberg-dark theme colors defined in `frontend/src/styles/theme.ts`
- ECharts used for other chart types (line, area, bar, heatmap)

## Tasks

### T1: Create /api/data/ticker/{ticker}/bars endpoint with timeframe support

**What:** New API endpoint that returns OHLCV bars for any cached timeframe.

**Files:**
- `api/routers/data_explorer.py` — add new endpoint

**Implementation notes:**
- Endpoint: `GET /api/data/ticker/{ticker}/bars?timeframe=15min&bars=500`
- Timeframe parameter: "5min", "15min", "30min", "1hr", "1d" (default: "1d")
- Look up cached parquet: `DATA_CACHE_DIR / f"{ticker}_{timeframe}_*.parquet"`
  - Use glob to find matching file regardless of date range in filename
- Return last N bars (default 500, max 5000):
  ```json
  {
    "ticker": "AAPL",
    "timeframe": "15min",
    "bars": [
      {"time": "2026-02-20T09:30:00", "open": 185.2, "high": 185.8, "low": 184.9, "close": 185.5, "volume": 1234567},
      ...
    ],
    "total_bars": 500,
    "available_timeframes": ["5min", "15min", "30min", "1hr", "1d"]
  }
  ```
- `available_timeframes` lists which timeframes have cached data for this ticker
- Handle missing timeframe gracefully: return 404 with message "No {timeframe} data cached for {ticker}"

**Verify:**
```bash
curl -s "http://localhost:8000/api/data/ticker/AAPL/bars?timeframe=5min&bars=100" | python -m json.tool | head -20
```

---

### T2: Create /api/data/ticker/{ticker}/indicators endpoint

**What:** API endpoint that computes technical indicators on cached OHLCV data and returns overlay/panel data for charting.

**Files:**
- `api/routers/data_explorer.py` — add indicator endpoint
- `indicators/indicators.py` — already has all indicator functions (read to understand interface)

**Implementation notes:**
- Endpoint: `GET /api/data/ticker/{ticker}/indicators?timeframe=15min&indicators=rsi_14,macd,bollinger_20`
- Parse indicator names and parameters from query string
- Available indicators (from indicators.py): RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, OBV, VWAP, EMA, SMA, Hurst, Ichimoku, etc.
- Return:
  ```json
  {
    "ticker": "AAPL",
    "timeframe": "15min",
    "indicators": {
      "rsi_14": {
        "type": "panel",
        "values": [{"time": "...", "value": 65.2}, ...],
        "thresholds": {"overbought": 70, "oversold": 30}
      },
      "bollinger_20": {
        "type": "overlay",
        "upper": [{"time": "...", "value": 186.5}, ...],
        "middle": [{"time": "...", "value": 185.0}, ...],
        "lower": [{"time": "...", "value": 183.5}, ...]
      }
    },
    "available_indicators": ["rsi", "macd", "bollinger", "atr", "adx", ...]
  }
  ```
- Indicator types: "overlay" (drawn on price chart) vs "panel" (separate panel below)
- Cache computed indicators for 60 seconds (avoid recomputing on every chart interaction)

**Verify:**
```bash
curl -s "http://localhost:8000/api/data/ticker/AAPL/indicators?timeframe=1d&indicators=rsi_14,macd" | python -m json.tool | head -30
```

---

### T3: Build enhanced CandlestickChart with timeframe selector and multi-pane layout

**What:** Upgrade the CandlestickChart component to support timeframe switching and indicator panels below the main chart.

**Files:**
- `frontend/src/components/charts/CandlestickChart.tsx` — major rewrite
- `frontend/src/components/charts/IndicatorPanel.tsx` — new component for RSI/MACD/etc. panels

**Implementation notes:**
- Use `lightweight-charts` for main candlestick pane (already installed)
- Timeframe selector: row of buttons [5m, 15m, 30m, 1H, 1D] — clicking fetches from T1 endpoint
- Main pane: candlestick + volume + overlay indicators (SMA, EMA, Bollinger)
- Sub-panes: stacked below main chart for panel indicators (RSI, MACD, Stochastic)
  - Each sub-pane is a separate `lightweight-charts` instance synced to same time axis
- Indicator selector: dropdown or sidebar with checkboxes for available indicators
- When user toggles an indicator, fetch from T2 endpoint, add series to appropriate pane
- Crosshair sync: when user hovers main chart, sub-pane crosshairs follow
- Volume bars: colored green/red based on close vs open
- Bloomberg-dark theme: use existing theme colors

**Verify:**
- Manual: Navigate to Data Explorer, select AAPL, verify timeframe buttons switch data, verify RSI appears in sub-pane below

---

### T4: Update DataExplorerPage with indicator sidebar and timeframe state

**What:** Redesign the Data Explorer page to include timeframe controls, indicator management, and data quality info per timeframe.

**Files:**
- `frontend/src/pages/DataExplorerPage.tsx` — update layout

**Implementation notes:**
- Layout:
  ```
  [Ticker Selector] [Timeframe: 5m 15m 30m 1H 1D] [Indicator +]
  ┌──────────────────────────────────────────────┐
  │  Main Candlestick Chart (with overlays)       │
  │  Volume bars                                  │
  ├──────────────────────────────────────────────┤
  │  RSI Panel (if enabled)                       │
  ├──────────────────────────────────────────────┤
  │  MACD Panel (if enabled)                      │
  └──────────────────────────────────────────────┘
  [Data Quality: 12,450 bars | Last: 2026-02-23 | Source: WRDS]
  ```
- State management:
  - `selectedTimeframe` in Zustand store (persists across ticker changes)
  - `enabledIndicators` in Zustand store
- When selecting a ticker, immediately check `available_timeframes` from T1 response
- Disable timeframe buttons that don't have cached data (gray out with tooltip "No data cached")
- Show "Last updated" timestamp from parquet metadata
- Show data source (WRDS, IBKR, yfinance) from cache metadata

**Verify:**
- Manual: Full Data Explorer workflow — select ticker, switch timeframes, add/remove indicators

---

### T5: Add indicator computation caching and batch loading

**What:** Optimize indicator loading so switching timeframes doesn't recompute everything.

**Files:**
- `api/routers/data_explorer.py` — add caching layer
- `api/cache/` — reuse existing cache infrastructure if available

**Implementation notes:**
- Cache key: `f"{ticker}_{timeframe}_{indicator}_{bar_count}"`
- TTL: 300 seconds (5 minutes) — intraday data doesn't change once cached
- Batch endpoint: `POST /api/data/ticker/{ticker}/indicators/batch` accepting list of indicators
  - Computes all requested indicators in one pass over the OHLCV data (more efficient than N separate requests)
- Pre-compute common indicators (RSI_14, MACD, Bollinger_20) when bars endpoint is called
  - Return `precomputed_indicators` field with the bars response for instant display

**Verify:**
```bash
# First call should compute, second should be cached (much faster)
time curl -s "http://localhost:8000/api/data/ticker/AAPL/indicators?timeframe=1d&indicators=rsi_14" > /dev/null
time curl -s "http://localhost:8000/api/data/ticker/AAPL/indicators?timeframe=1d&indicators=rsi_14" > /dev/null
```

---

### T6: Add TradingView reference widget as optional panel

**What:** Embed the free TradingView Advanced Chart widget as an optional "Market Reference" panel that shows TradingView's real-time data alongside the custom data chart.

**Files:**
- `frontend/src/components/charts/TradingViewWidget.tsx` — new component
- `frontend/src/pages/DataExplorerPage.tsx` — add toggle for TV widget

**Implementation notes:**
- Use the free TradingView embed widget (no license needed):
  ```tsx
  const TradingViewWidget = ({ symbol, theme = "dark", interval = "D" }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
      const script = document.createElement("script");
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.async = true;
      script.innerHTML = JSON.stringify({
        symbol: symbol,
        theme: theme,
        interval: interval,
        style: "1",
        timezone: "America/New_York",
        hide_side_toolbar: false,
        allow_symbol_change: true,
        studies: ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
      });
      containerRef.current?.appendChild(script);
      return () => { containerRef.current?.innerHTML = ""; };
    }, [symbol, interval]);
    return <div ref={containerRef} style={{ height: "500px" }} />;
  };
  ```
- Toggle button: "Show TradingView Reference" — opens TV widget in collapsible panel below custom chart
- Sync symbol: when user selects a ticker in the data explorer, update TV widget symbol
- This is supplementary — primary chart remains the custom lightweight-charts with user's data

**Verify:**
- Manual: Click "Show TradingView Reference", verify TradingView chart loads with same ticker

---

### T7: Test charting endpoints and data flow

**What:** Unit and integration tests for the new bars and indicator endpoints.

**Files:**
- `tests/test_charting_endpoints.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_bars_endpoint_daily` — /bars?timeframe=1d returns valid OHLCV
  2. `test_bars_endpoint_intraday` — /bars?timeframe=15min returns valid data if cached
  3. `test_bars_endpoint_missing_timeframe` — Returns 404 with helpful message
  4. `test_available_timeframes` — Response includes correct list of cached timeframes
  5. `test_indicator_endpoint_rsi` — /indicators?indicators=rsi_14 returns correct structure
  6. `test_indicator_endpoint_overlay_vs_panel` — Bollinger returns "overlay" type, RSI returns "panel" type
  7. `test_indicator_caching` — Second request is faster than first
  8. `test_batch_indicators` — POST endpoint computes multiple indicators in one call

**Verify:**
```bash
python -m pytest tests/test_charting_endpoints.py -v
```

---

## Validation

### Acceptance criteria
1. Data Explorer shows candlestick chart with working timeframe selector (5m, 15m, 30m, 1H, 1D)
2. At least 3 indicators can be overlaid (RSI, MACD, Bollinger) with correct visualization
3. Intraday data loads from cached parquets (not re-downloaded)
4. TradingView reference widget loads when toggled
5. Switching tickers updates both custom chart and TV widget
6. Volume bars are colored green/red
7. Charts use Bloomberg-dark theme

### Rollback plan
- Keep existing CandlestickChart.tsx as `CandlestickChartLegacy.tsx`
- New endpoints are additive — old `/data/ticker/{ticker}` still works
- TradingView widget is optional toggle — remove component if issues
