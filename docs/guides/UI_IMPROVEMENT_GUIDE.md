# Quant Engine UI: Complete Improvement Guide

**Date:** February 21, 2026
**Current State:** 12 files, ~4,700 lines, tkinter + matplotlib desktop app
**Pages:** 8 (Dashboard, Backtest/Risk, Model Lab, Signal Desk, Data Explorer, IV Surface, Autopilot/Kalshi, S&P Comparison)

---

## THE CORE DECISION: tkinter vs Web

Your current UI is built on tkinter with embedded matplotlib. This works, and the Bloomberg-inspired dark theme looks professional. But tkinter has hard ceilings that no amount of polish can fix: static charts (no hover, no zoom, no pan), desktop-only (can't share via URL), no real-time streaming, and matplotlib gets sluggish beyond a few thousand data points.

You have two paths:

**Path A: Polish the tkinter app** — faster, keeps what you have, good enough for personal use. Focus on connecting real data, adding interactivity where tkinter allows, and fixing UX gaps.

**Path B: Rewrite as a web app** — more work upfront, but gives you interactive Plotly charts, real-time updates via WebSocket, accessible from any browser, deployable to a server. This is what production trading desks use.

Both paths are laid out below. Most of the improvements in Sections 1-3 apply to either path. Section 4 is tkinter-specific. Section 5 is the web rewrite blueprint.

---

## SECTION 1: CONNECT REAL DATA (Highest Priority)

Right now, most pages generate synthetic/demo data. The backend is fully built — backtest engine, model registry, feature pipeline, risk modules — but the UI doesn't call any of it. This is the single biggest improvement you can make regardless of framework.

### 1.1 Dashboard → Real Backtest Results

**Current:** Dashboard loads from `results/backtest_trades.csv` and `trained_models/registry.json` with extensive demo fallbacks.

**What to wire:**
- Equity curve: load from `results/backtest_10d_summary.json` + compute cumulative returns from trade CSV
- Regime state: call `RegimeDetector.regime_features()` on latest data, show current HMM state + probabilities
- Model health: read `trained_models/ensemble_10d_meta.json` for CV gap, holdout R², IC values
- Feature importance: already partially wired (reads model meta JSON) — ensure path is correct
- Trade log: read actual `results/backtest_trades.csv` if it exists
- Risk metrics: compute from trade returns using `risk/metrics.py` functions

### 1.2 Backtest/Risk → Real Engine Integration

**Current:** "Run Backtest" button sleeps 1.5 seconds and shows demo data.

**What to wire:**
- Import and call `BacktestEngine.run()` with the parameters from the UI controls (holding period, max position, entry threshold)
- Display actual equity curve, drawdown, monthly returns from the engine output
- Use `risk/attribution.py` for the attribution breakdown
- Use `risk/stress_test.py` results for the risk analytics tab
- This is the most complex wiring job — the backtest engine takes a predictions DataFrame and returns a results dict

### 1.3 Model Lab → Real Training

**Current:** "Train Model" button shows a fake progress bar for 2 seconds.

**What to wire:**
- Import `ModelTrainer` and call `train()` in a background thread
- Stream CV fold results back to the UI as they complete (use a queue between the training thread and the UI thread)
- Show actual feature importance from the trained model
- Display real regime detection results from `RegimeDetector`

### 1.4 Signal Desk → Real Predictions

**Current:** All demo signals.

**What to wire:**
- Import `EnsemblePredictor` and generate predictions on latest data
- Show actual predicted returns, confidence scores, regime labels
- Use `CrossSectionalRanker` for the ranking

### 1.5 Autopilot/Kalshi → Real Strategy Discovery

**Current:** All demo.

**What to wire:**
- Import `AutopilotEngine` and run a discovery cycle
- Display actual candidates, promotion results from `results/autopilot/latest_cycle.json`
- Show real paper trading state from the paper trader's state file

---

## SECTION 2: MAKE CHARTS INTERACTIVE

This is the second biggest impact improvement. Static matplotlib charts with no hover, no zoom, and no data inspection are the main thing that makes the UI feel like a prototype instead of a product.

### Option A: Plotly in tkinter (Hybrid Approach)

You can embed Plotly charts in tkinter using a lightweight local web view. This keeps tkinter as the shell but gives you interactive charts.

**How it works:**
- Render Plotly figures to HTML
- Display in a tkinter `tkinterweb.HtmlFrame` or `cefpython3` embedded browser
- Charts get hover tooltips, zoom, pan, export-to-PNG for free

**Pros:** Keep existing app structure, add interactivity incrementally
**Cons:** Adds dependency (tkinterweb or cefpython3), slight performance overhead

### Option B: Replace matplotlib with Plotly throughout

Rewrite `charts.py` to produce Plotly figures instead of matplotlib. Each chart becomes an HTML file rendered in an embedded browser widget.

**What changes:**
- `create_line_chart()` → returns `plotly.graph_objects.Figure` instead of `matplotlib.figure.Figure`
- `embed_figure()` → writes figure to temp HTML, loads in embedded browser
- All chart styling moves from matplotlib rcParams to Plotly template

### Option C: Web rewrite (see Section 5)

Move charts to the browser entirely where Plotly/D3 are native.

### Specific Chart Improvements Regardless of Library

**Equity curves:** Add drawdown shading as a second y-axis, hover showing exact date/value/drawdown
**Regime timeline:** Make regime bands clickable — clicking a regime shows that period's stats
**Heatmaps:** Add cell hover with exact values, click to drill into a specific cell
**3D IV surface:** Add rotation/zoom (Plotly 3D does this natively; matplotlib's is clunky)
**Trade scatter:** Color by P&L, size by position size, hover shows full trade details
**Monthly returns:** Click a month to see daily breakdown

---

## SECTION 3: UX IMPROVEMENTS (Framework-Agnostic)

### 3.1 Loading States

**Current:** Status label says "Loading..." with no visual feedback.

**What to add:**
- Indeterminate progress bar while data loads (tkinter has `ttk.Progressbar(mode='indeterminate')`)
- Determinate progress for multi-step operations (training: show fold 1/5, 2/5, etc.)
- Skeleton screens: show grey placeholder rectangles where charts will appear before data loads
- Disable buttons during operations (prevent double-clicks)

### 3.2 Keyboard Shortcuts

**What to add:**
- `Ctrl+R` / `Cmd+R` → Refresh current page
- `Ctrl+1` through `Ctrl+8` → Jump to page 1-8
- `Ctrl+E` → Export current view
- `Escape` → Cancel running operation
- `Ctrl+F` → Find in trade log / data table
- Show shortcuts in a help overlay (`Ctrl+?` or `F1`)

### 3.3 Data Export

**What to add:**
- "Export" button on every chart → saves PNG (matplotlib: `fig.savefig()`)
- "Export CSV" button on every table → saves the underlying DataFrame
- "Export Report" → generates a PDF summary of the current page (use `risk/attribution.py` output + chart images)
- Copy-to-clipboard for metric values (click a metric card to copy its value)

### 3.4 Table Improvements

**Current:** tkinter Treeview with basic columns, no sorting, no filtering.

**What to add:**
- Click column header to sort (ascending/descending toggle)
- Right-click context menu (copy row, export selection, view details)
- Search/filter bar above tables
- Alternating row colors for readability
- Pagination for large tables (show 50 per page with page controls)
- Column resize handles

### 3.5 Better Navigation

**What to add:**
- Breadcrumbs showing current location (e.g., "Dashboard > Regime State")
- Recent items shortcut (last 5 visited pages)
- Quick-jump search bar (type "regime" to jump to regime tab)
- Page refresh button in the header bar

### 3.6 Tooltips and Help

**What to add:**
- Hover tooltips on metric cards explaining what each metric means (e.g., "CV Gap: difference between in-sample and out-of-sample performance. Lower is better. >0.10 indicates overfitting.")
- "?" icons next to complex controls that open an explanation popup
- First-time setup wizard that walks through connecting data sources

### 3.7 Alerts Panel

**What to add:**
- A notification area (top-right bell icon or dedicated sidebar section)
- Alert types: model retrain needed, drawdown threshold breached, regime change detected, data quality issue
- Alert history with timestamps
- Click alert to jump to relevant page

### 3.8 Settings Page

**What to add:**
- Theme toggle (dark/light) — your theme.py already defines colors; add a light variant
- Auto-refresh interval (currently hardcoded 30s in dashboard)
- Default universe selection
- Data paths configuration
- Export format preferences (PNG resolution, CSV delimiter)
- Alert thresholds (configurable drawdown limit, IC threshold, etc.)

---

## SECTION 4: tkinter-SPECIFIC IMPROVEMENTS

If you stay with tkinter, these are the highest-impact changes.

### 4.1 Fix Memory Management

**Problem:** Pages are cached forever in `self._pages` dict. Matplotlib figures are never explicitly closed. Over a long session, memory grows.

**Fix:**
- Add `fig.clf()` and `plt.close(fig)` when a chart is replaced
- Implement a page eviction policy (e.g., only keep the 3 most recently visited pages in memory)
- Add a "Clear Cache" button in settings

### 4.2 Improve Chart DPI for Retina Displays

**Current:** 100 DPI everywhere.

**Fix:**
- Detect display scaling: `root.tk.call('tk', 'scaling')` returns the current scale factor
- Set DPI to 100 × scale_factor (e.g., 200 DPI on Retina)
- This makes charts crisp on high-DPI screens without changing layout

### 4.3 Add Chart Toolbar

**Current:** No interaction with charts.

**Fix:**
- Add matplotlib's `NavigationToolbar2Tk` below each chart figure
- This gives you zoom, pan, home, save-to-PNG buttons for free
- One line per chart: `NavigationToolbar2Tk(canvas, parent_frame)`
- Not as good as Plotly interactivity, but dramatically better than nothing

### 4.4 Add Right-Click Context Menus

**Current:** No context menus anywhere.

**Fix:**
- Add `tk.Menu(tearoff=0)` popup menus on Treeview rows, chart canvases, and metric cards
- Treeview: "Copy Row", "View Details", "Export Selection"
- Charts: "Save as PNG", "Copy to Clipboard", "Reset Zoom"
- Metric cards: "Copy Value", "Show History"

### 4.5 Add Tab Keyboard Navigation

**Current:** Tabs are click-only.

**Fix:**
- Bind `Left`/`Right` arrow keys to switch tabs within a notebook
- Bind `Ctrl+Tab` to cycle through main pages
- Show the active tab with a more prominent indicator

### 4.6 Improve the Status Bar

**Current:** Shows Python version and datetime. Not very useful.

**Replace with:**
- Last data refresh time
- Current regime state (color-coded)
- Model age (days since last training)
- Active alerts count (with color badge)
- Memory usage

### 4.7 Add a Splash Screen

**Current:** App opens with a blank window while pages initialize.

**Fix:**
- Show a branded splash screen (logo + "Loading Quant Engine..." + progress bar) for 1-2 seconds
- Initialize the most-used page (Dashboard) during splash
- Lazy-load everything else

---

## SECTION 5: WEB REWRITE BLUEPRINT

If you decide to go web-based, here's the architecture.

### Technology Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Backend | **FastAPI** (Python) | Async, fast, easy to integrate with existing Python codebase |
| Frontend | **React + TypeScript** | Component-based, massive ecosystem, fast |
| Charts | **Plotly.js** or **Lightweight Charts** (TradingView) | Interactive, financial-grade |
| State | **Zustand** or **React Query** | Lightweight, handles caching + real-time |
| Styling | **Tailwind CSS** | Rapid dark-theme development |
| Real-time | **WebSocket** (via FastAPI) | Live portfolio updates, regime changes |
| Build | **Vite** | Fast dev server, instant HMR |

### Backend API Structure

The backend wraps your existing Python modules:

```
GET  /api/portfolio/summary      → calls risk/metrics.py
GET  /api/portfolio/equity-curve  → reads backtest results
GET  /api/regime/current          → calls RegimeDetector
GET  /api/model/health            → reads model registry JSON
GET  /api/signals/latest          → calls EnsemblePredictor
GET  /api/trades/recent           → reads backtest trades CSV
POST /api/backtest/run            → calls BacktestEngine
POST /api/model/train             → calls ModelTrainer
WS   /ws/live                     → streams regime changes, alerts
```

### Frontend Page Layout

Same 8 pages, but as React components with Plotly charts:

```
<App>
  <Sidebar />          // Same 8 items, keyboard navigable
  <Header />           // Alerts bell, settings gear, search bar
  <MainContent>
    <Dashboard />      // Plotly equity curve, regime timeline, metric cards
    <BacktestRisk />   // Interactive drawdown, monthly heatmap
    <ModelLab />       // Feature importance, training progress
    <SignalDesk />     // Signal table with sorting/filtering
    <DataExplorer />   // Candlestick charts with TradingView widget
    <IVSurface />      // 3D Plotly surface with rotation
    <Autopilot />      // Strategy funnel, paper trading
    <SPComparison />   // Dual equity curves with animation
  </MainContent>
  <StatusBar />        // Regime indicator, model age, alerts
</App>
```

### What You Gain from Web

- **Interactive charts** with hover, zoom, pan, crosshair, export
- **Real-time updates** via WebSocket (regime changes, new trades, alerts push instantly)
- **Accessible anywhere** — open in any browser, share URL with collaborators
- **Responsive** — works on laptop, tablet, second monitor
- **TradingView-quality candlestick charts** via Lightweight Charts library
- **Server-side computation** — heavy operations (backtest, training) run on server, UI stays responsive
- **Authentication** — add login if you ever want multi-user
- **PDF report generation** — server renders reports, sends to browser for download

### Migration Strategy

Don't rewrite everything at once. Migrate page by page:

1. Start with Dashboard (highest value, most data integration)
2. Add Data Explorer (easiest to make interactive — candlestick charts)
3. Add IV Surface (3D Plotly is dramatically better than matplotlib 3D)
4. Add Signal Desk (table sorting/filtering is trivial in React)
5. Add Backtest/Risk (needs backend API for backtest engine)
6. Add remaining pages

Keep the tkinter app running in parallel during migration. Both can read from the same data files.

---

## SECTION 6: VISUAL POLISH (Quick Wins)

These are small changes that make the app feel more professional regardless of framework.

### 6.1 Metric Card Animations

When a metric card value changes (e.g., on refresh), briefly flash the card border green (if improved) or red (if degraded). This gives visual feedback that data actually updated.

### 6.2 Chart Color Consistency

Ensure the same ticker always gets the same color across all charts. Create a deterministic color mapping: `color = PALETTE[hash(ticker) % len(PALETTE)]`.

### 6.3 Better Number Formatting

Your `charts.py` has `format_pct()` and `format_number()`. Extend these:
- Large numbers: `1,234,567` → `$1.23M`
- Sharpe ratios: always show 2 decimal places
- Percentages: always show sign (+3.2%, -1.5%)
- Dates: use relative ("2 hours ago", "yesterday") for recent, absolute for old

### 6.4 Empty State Design

When a page has no data (first run, missing files), show a helpful empty state instead of a blank screen:
- "No backtest results found. Run a backtest from the Backtest & Risk page."
- Include a button that navigates to the relevant page
- Use a subtle illustration or icon

### 6.5 Consistent Tab Styling

Some pages use `ttk.Notebook` tabs, others use custom buttons. Standardize on one approach with consistent sizing, padding, and active indicators.

### 6.6 Chart Titles and Labels

Every chart should have: a clear title (what am I looking at?), axis labels with units (% for returns, $ for values, dates for time), and a legend if multiple series. Some of your charts have this; make it universal.

---

## PRIORITY SUMMARY

| Priority | Improvement | Impact | Effort |
|----------|-------------|--------|--------|
| **1** | Connect real data to all pages (Section 1) | Transforms from demo to functional | Medium |
| **2** | Add matplotlib toolbar to all charts (4.3) | Instant zoom/pan/save | Low |
| **3** | Add loading states and progress bars (3.1) | Feels responsive | Low |
| **4** | Add keyboard shortcuts (3.2) | Power user efficiency | Low |
| **5** | Add data export buttons (3.3) | Professional utility | Low |
| **6** | Fix memory management (4.1) | Prevents crashes in long sessions | Low |
| **7** | Table sorting and filtering (3.4) | Usability for trade logs | Medium |
| **8** | Add tooltips on metrics (3.6) | Reduces confusion | Low |
| **9** | Add alerts panel (3.7) | Proactive monitoring | Medium |
| **10** | Add settings page (3.8) | User customization | Medium |
| **11** | Retina DPI fix (4.2) | Crisp visuals on Mac | Low |
| **12** | Replace matplotlib with Plotly (Section 2) | Interactive charts | High |
| **13** | Web rewrite (Section 5) | Full modernization | Very High |
