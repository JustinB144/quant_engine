# IBKR + WRDS Data Integration Report

**Date:** February 22, 2026
**Scope:** Full integration of IBKR intraday data + WRDS daily data into quant engine pipeline
**Cache Location:** `quant_engine/data/cache/`

---

## 1. Current State of Cache

### 1.1 File Inventory

| Metric | Value |
|--------|-------|
| Total files | 910 |
| Parquet files | 455 |
| CSV files | 455 (exact duplicates of parquet) |
| Unique tickers | 154 |
| Metadata sidecar files (.meta.json) | 0 |
| Total cache size | 424 MB |

### 1.2 Data Sources

| Timeframe | Source | File Count (parquet) | Tickers | Date Range |
|-----------|--------|---------------------|---------|------------|
| daily | WRDS CRSP | 155 | 154 + SPY | 2006-02-21 → 2024-12-31 |
| 4hour | IBKR | 100 | 100 | 2015-12-14 → 2026-02-20 |
| 1hour | IBKR | 100 | 100 | 2015-10-12 → 2026-02-20 |
| 30min | IBKR | 100 | 100 | 2020-11-20 → 2026-02-20 |

### 1.3 Bar Counts (AAPL Sample)

| Timeframe | Bars | Years of Data |
|-----------|------|---------------|
| daily | 4,748 | ~19 |
| 4hour | 6,011 | ~10 |
| 1hour | 18,166 | ~10 |
| 30min | 17,049 | ~5 |

### 1.4 File Naming Convention

All files use this IBKR export pattern (including the WRDS daily files):
```
{TICKER}_{timeword}_{start_date}_{end_date}.parquet
{TICKER}_{timeword}_{start_date}_{end_date}.csv
```

Where `{timeword}` is one of: `daily`, `4hour`, `1hour`, `30min`

Examples:
```
AAPL_daily_2006-02-21_2024-12-31.parquet
AAPL_4hour_2015-12-14_2026-02-20.parquet
AAPL_1hour_2015-10-12_2026-02-20.parquet
AAPL_30min_2020-11-20_2026-02-20.parquet
```

### 1.5 Column Format

All files have identical column structure:
```csv
Date,Open,High,Low,Close,Volume
```

Daily timestamps are date-only:
```
2006-02-21,69.08,69.08,69.08,69.08,27905392.0
```

Intraday timestamps are timezone-aware:
```
2015-12-14 09:30:00-05:00,28.05,28.05,27.65,27.79,79536396.0
```

---

## 2. Issues Found

### ISSUE 1 — CRITICAL: Daily Data Has Identical Open/High/Low/Close

**Severity:** CRITICAL — breaks ~40% of all computed features
**Affected:** ALL 155 daily files (every ticker, every row)
**Root Cause:** WRDS CRSP `dsf` table by design

Every daily bar has Open = High = Low = Close. This is NOT a download error. The WRDS provider at `wrds_provider.py:478-482` explicitly sets this:

```python
ohlcv = pd.DataFrame({
    'Open':   g['prc'],   # CRSP has Close only; set O=H=L=C
    'High':   g['prc'],
    'Low':    g['prc'],
    'Close':  g['prc'],
    'Volume': g['vol'].fillna(0),
```

The CRSP Daily Stock File (`crsp.dsf`) only carries `prc` (closing price), `ret` (return), and `vol` (volume). It does not contain open, high, or low prices. The `openprc`, `askhi`, and `bidlo` fields exist in CRSP but are in the `crsp.dsf` table only for certain date ranges and are not being queried.

**Features broken by O=H=L=C (all produce zero, NaN, or degenerate values):**

- ATR / NATR — `max(H-L, |H-prevC|, |L-prevC|)` collapses since H=L
- Parkinson Volatility — `log(H/L) = log(1) = 0` always
- Garman-Klass Volatility — uses `(H-L)^2` which is 0
- Yang-Zhang Volatility — uses all four OHLC, produces 0
- Stochastic Oscillator / Williams %R — `(C-L)/(H-L)` = 0/0 = NaN
- CCI — typical price `(H+L+C)/3` degenerates to just `C`
- Candlestick Body / Direction — `Close - Open = 0` always
- VWAP Bands — typical price collapses
- Value Area High/Low — no range to compute
- ATR Trailing Stop / ATR Channel — ATR = 0
- Range Breakout — no range
- Pivot High / Pivot Low — all pivots identical
- HigherHighs / LowerLows — never triggered
- Gap Percent — Open always equals previous Close in the data
- Bollinger Band Width — partially affected (uses Close std, but BBW percentile context is wrong)
- Volatility Squeeze — depends on ATR which is 0

**Features that still work correctly** (Close/Volume/Return only):

RSI, MACD, MACD Signal/Histogram, ROC, SMA, EMA, Price vs SMA, SMA Slope, ADX (partially — DI uses H/L), OBV, OBV Slope, Amihud Illiquidity, Kyle Lambda, Roll Spread, Hurst Exponent, Z-Score, Variance Ratio, Autocorrelation, Kalman Trend, Shannon Entropy, Approximate Entropy, Fractal Dimension, DFA, Return Skewness, Return Kurtosis, CUSUM Detector, all GARCH-based features, Historical Volatility (close-to-close), Volume Ratio, MFI (partially), RVOL

**Fix required:** Modify `wrds_provider.py:get_crsp_prices()` to query `openprc`, `askhi`, `bidlo` from CRSP alongside `prc`, then re-download. Alternatively, use IBKR daily data which has proper OHLCV — but IBKR only goes back ~10 years vs WRDS's 20.

### ISSUE 2 — CRITICAL: File Naming Does Not Match Primary Loader Path

**Severity:** CRITICAL — parquet files silently ignored, falls through to slower CSV
**Affected:** ALL daily files, both loading and metadata generation

The primary loader `load_ohlcv_with_meta()` at `local_cache.py:220` looks for an exact path:
```python
parquet_path = root / f"{ticker.upper()}_1d.parquet"
```

But files are named `AAPL_daily_2006-02-21_2024-12-31.parquet`. There is NO glob for `*_daily_*.parquet` in the load path.

The CSV fallback at `local_cache.py:108` DOES glob for `{t}_daily_*.csv`:
```python
candidates.extend(sorted(cache_root.glob(f"{t}_daily_*.csv")))
```

So daily data loads successfully, but ONLY via the CSV path (slower, larger files). The parquet files — which are faster to read and ~40-60% smaller — are completely unused.

Additionally, `_daily_cache_files()` at `local_cache.py:298-311` and `list_cached_tickers()` at `local_cache.py:281-295` both have the same gap — they glob for `*_daily_*.csv` but NOT `*_daily_*.parquet`.

### ISSUE 3 — CRITICAL: No Metadata Sidecar Files Exist

**Severity:** CRITICAL — all data treated as untrusted, triggers unnecessary WRDS/yfinance re-fetches
**Affected:** ALL 455 parquet + 455 CSV files

Zero `.meta.json` files exist in the cache. The loader checks metadata source against `CACHE_TRUSTED_SOURCES = ["wrds", "wrds_delisting", "ibkr"]` at `config.py:54`. Without metadata, the source defaults to `"unknown"`, which means:

1. `_cache_is_usable()` with `require_trusted=True` returns `False` at `loader.py:97-98`
2. Loader skips trusted cache path at `loader.py:266` and falls through to WRDS or yfinance
3. If WRDS credential is available, it re-queries CRSP on every run (slow, unnecessary)
4. If WRDS is unavailable, it falls to yfinance, which may overwrite your data with lower-quality yfinance data

The `run_rehydrate_cache_metadata.py` script exists specifically for this purpose, but it also has a parquet gap — `_daily_cache_files()` at line 303 only scans `*_daily_*.csv`, not `*_daily_*.parquet`. It also only processes daily files, not intraday.

### ISSUE 4 — HIGH: Old WRDS PERMNO-Keyed Files No Longer Exist

**Severity:** HIGH — benchmark loading and PERMNO-based lookups broken
**Affected:** SPY benchmark, any code using PERMNO as cache key

The old files `84398_1d.parquet` (SPY by PERMNO) and associated `.meta.json` sidecars no longer exist. SPY is now at `SPY_daily_2006-02-21_2024-12-31.parquet`. Multiple code paths are affected:

- `dash_ui/data/loaders.py` benchmark loader tries `SPY_1d.parquet` then `84398_1d.parquet`
- `local_cache.py:239` alias lookup scans `*_1d.meta.json` files — none exist
- Any PERMNO-keyed load (e.g., `load_ohlcv("84398")`) will fail since the metadata mapping is gone

### ISSUE 5 — HIGH: Intraday Data Not Accessible to Feature Pipeline

**Severity:** HIGH — 13 features remain NaN even with data available
**Affected:** `intraday.py` (6 features) + `lob_features.py` (7 features)

The feature pipeline at `pipeline.py:700-745` only attempts intraday/LOB feature computation when WRDS TAQmsec is available. There is no code path to read local IBKR intraday files from the cache. Your 100-ticker × 3-timeframe intraday dataset is completely unused by the feature pipeline.

The features that would be unlocked:

From `lob_features.py` (computable from 30min/1hr bars):
- `trade_arrival_rate` — Poisson lambda from inter-bar durations
- `quote_update_intensity` — price changes per unit time
- `duration_between_trades_mean` — mean inter-trade duration
- `duration_between_trades_std` — std of inter-trade durations
- `price_impact_asymmetry` — buy vs sell impact
- `queue_imbalance` — ratio of up-moves to down-moves
- `fill_probability_proxy` — fraction of bars where volume > median

From `intraday.py` (requires 1-minute bars — NOT YET AVAILABLE):
- `intraday_vol_ratio` — first-hour / last-hour volume
- `vwap_deviation` — (close - VWAP) / VWAP
- `amihud_illiquidity` — mean(|ret| / dollar_volume) × 1e6
- `kyle_lambda` — OLS slope of signed_sqrt(vol) on price change
- `realized_vol_5m` — annualized realized vol from 5-min returns
- `microstructure_noise` — excess variance ratio (Hansen & Lunde)

### ISSUE 6 — HIGH: Daily Data Ends Dec 2024, Intraday Extends to Feb 2026

**Severity:** HIGH — 14-month gap in daily features for recent dates
**Affected:** 138+ of 155 daily files

Most daily files end at `2024-12-31`. Intraday data runs through `2026-02-20`. This means for the most recent 14 months, the pipeline has no daily data to compute features from, even though intraday data exists. A few daily files extend to `2026-02-20` (likely re-downloaded from a different source).

### ISSUE 7 — MEDIUM: 28 UNIVERSE_FULL Tickers Missing from Cache

**Severity:** MEDIUM — reduced universe coverage
**Affected:** 28 of 53 tickers in `config.py:UNIVERSE_FULL`

These tickers have no data at any timeframe:
```
AMD, INTC, CRM, ADBE, ORCL       (large-cap tech)
DDOG, NET, CRWD, ZS, SNOW,       (mid-cap tech/cyber)
MDB, PANW, FTNT                   (mid-cap tech/cyber)
ABBV, TMO                         (healthcare)
MCD, TGT, COST                    (consumer)
GS, BLK, MA                       (financial)
BA                                (industrial)
CAVA, BROS, TOST, CHWY, ETSY, POOL  (volatile/small-mid)
```

### ISSUE 8 — MEDIUM: Duplicate CSV + Parquet for Every File

**Severity:** MEDIUM — wastes ~200MB disk space, no functional impact
**Affected:** All 455 ticker-timeframe combinations

Every file exists in both CSV and parquet. Parquet is preferred (faster reads, smaller files, preserves dtypes). After confirming parquet loading works, CSVs can be deleted.

### ISSUE 9 — LOW: 54 Tickers Have Daily Only (No Intraday)

**Severity:** LOW — these tickers get no intraday features but otherwise function
**Affected:** 54 tickers

```
AEE, AFL, AIG, ALL, AMP, APD, BALL, BAX, BEN, CAG, CAH, CMS, CNP,
DTE, FISV, GLW, HAS, HBAN, HPQ, IP, JCI, KEY, L, LIN, MAS, MET,
MRSH, MTB, NTRS, OMC, PAYX, PFG, PHM, PNW, PPG, PRU, PSKY, RVTY,
SPGI, SRE, STT, STZ, T, TAP, TPR, TROW, TRV, TSN, TXT, VRSN,
VTRS, WM, WMB, ZBH
```

---

## 3. Required Code Changes (Detailed)

### 3.1 `data/wrds_provider.py` — Fix CRSP OHLCV Query

**File:** `data/wrds_provider.py`
**Function:** `get_crsp_prices()` starting at line 406
**Lines to change:** 445-482

The CRSP `dsf` table does carry `openprc`, `askhi`, `bidlo` for a subset of records. However, these fields have limited coverage (not all stocks, not all dates). The correct fix is a two-part approach:

**Part A:** Update the SQL query at line 446 to also select `openprc`, `askhi`, `bidlo`:
```sql
SELECT a.permno, a.date, a.prc, a.openprc, a.askhi, a.bidlo,
       a.ret, a.vol, a.shrout, b.ticker, b.comnam
FROM crsp.dsf AS a
```

**Part B:** Update the DataFrame construction at lines 478-482 to use real OHLCV when available, falling back to `prc` when not:
```python
ohlcv = pd.DataFrame({
    'Open':   g['openprc'].fillna(g['prc']),
    'High':   g['askhi'].fillna(g['prc']),
    'Low':    g['bidlo'].fillna(g['prc']),
    'Close':  g['prc'],
    'Volume': g['vol'].fillna(0),
    ...
})
```

This gives true OHLCV when CRSP has it and gracefully degrades to close-only when it doesn't.

**After this fix:** Re-run the WRDS data download for all tickers to refresh the daily cache files with proper OHLCV values.

### 3.2 `data/local_cache.py` — Fix Parquet Loading for New Naming Convention

**File:** `data/local_cache.py`
**Functions to modify:** `load_ohlcv_with_meta()` (line 207), `_daily_cache_files()` (line 298), `list_cached_tickers()` (line 281), `_cache_meta_path()` (line 121)

**3.2a: `load_ohlcv_with_meta()` — Add parquet glob fallback**

After the exact `_1d.parquet` check at line 220, add a glob for `*_daily_*.parquet`:

```python
# After line 229 (end of the exact parquet_path block), add:
parquet_daily_globs = sorted(root.glob(f"{ticker.upper()}_daily_*.parquet"))
for parquet_path in parquet_daily_globs:
    try:
        raw = pd.read_parquet(parquet_path)
        normalized = _to_daily_ohlcv(raw)
        if normalized is not None:
            meta = _read_cache_meta(parquet_path, ticker)
            return normalized, meta, parquet_path
    except Exception:
        pass
```

**3.2b: `_daily_cache_files()` — Add parquet glob**

Line 303 only scans CSV. Add parquet scanning:

```python
def _daily_cache_files(root: Path) -> List[Path]:
    candidates: List[Path] = []
    candidates.extend(sorted(root.glob("*_1d.parquet")))
    candidates.extend(sorted(root.glob("*_daily_*.parquet")))  # ADD THIS
    candidates.extend(sorted(root.glob("*_1d.csv")))
    candidates.extend(sorted(root.glob("*_daily_*.csv")))
    # ... rest unchanged
```

**3.2c: `list_cached_tickers()` — Add parquet glob**

Line 289-294 only globs `*_1d.parquet` and `*_daily_*.csv`. Add `*_daily_*.parquet`:

```python
for p in root.glob("*_daily_*.parquet"):  # ADD THIS
    tickers.add(p.stem.split("_daily_")[0].upper())
```

**3.2d: `_cache_meta_path()` — Handle new naming**

Line 121-125 only recognizes `_1D.` prefix. Update to also handle `_daily_`:

```python
def _cache_meta_path(data_path: Path, ticker: str) -> Path:
    canonical = data_path.parent / f"{ticker.upper()}_1d.meta.json"
    if data_path.name.upper().startswith(f"{ticker.upper()}_1D."):
        return data_path.with_suffix(".meta.json")
    if "_DAILY_" in data_path.name.upper() or "_daily_" in data_path.name:
        return data_path.with_suffix(".meta.json")
    return canonical
```

### 3.3 `data/local_cache.py` — Add Intraday Load Function

**New function** to add after `load_ohlcv()` at line 279:

```python
_IBKR_TIMEFRAME_MAP = {
    "4h": "4hour", "1h": "1hour", "30m": "30min",
    "15m": "15min", "5m": "5min", "1m": "1min",
}

def load_intraday_ohlcv(
    ticker: str,
    timeframe: str,
    cache_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load intraday OHLCV data from cache.

    Handles IBKR naming: {TICKER}_{timeword}_{start}_{end}.parquet
    Does NOT apply _to_daily_ohlcv() (which rejects sub-daily data).

    Args:
        ticker: Stock symbol (e.g. "AAPL")
        timeframe: Canonical code ("4h", "1h", "30m", "15m", "5m", "1m")
        cache_dir: Override cache directory

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns, or None
    """
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    t = ticker.upper()
    ibkr_tf = _IBKR_TIMEFRAME_MAP.get(timeframe, timeframe)

    # Try parquet first (faster)
    for pattern in [f"{t}_{ibkr_tf}_*.parquet", f"{t}_{timeframe}_*.parquet",
                    f"{t}_{timeframe}.parquet"]:
        matches = sorted(d.glob(pattern))
        for path in matches:
            try:
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors="coerce")
                df = _normalize_ohlcv_columns(df)
                if set(REQUIRED_OHLCV).issubset(df.columns):
                    df = df[REQUIRED_OHLCV].copy()
                    df = df.sort_index()
                    df = df[~df.index.isna()]
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.dropna(subset=REQUIRED_OHLCV)
                    if len(df) > 0:
                        return df
            except Exception:
                continue

    # CSV fallback
    for pattern in [f"{t}_{ibkr_tf}_*.csv", f"{t}_{timeframe}_*.csv",
                    f"{t}_{timeframe}.csv"]:
        matches = sorted(d.glob(pattern))
        for path in matches:
            try:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors="coerce")
                df = _normalize_ohlcv_columns(df)
                if set(REQUIRED_OHLCV).issubset(df.columns):
                    df = df[REQUIRED_OHLCV].copy()
                    df = df.sort_index()
                    df = df[~df.index.isna()]
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.dropna(subset=REQUIRED_OHLCV)
                    if len(df) > 0:
                        return df
            except Exception:
                continue
    return None


def list_intraday_timeframes(
    ticker: str,
    cache_dir: Optional[Path] = None,
) -> List[str]:
    """Return list of available intraday timeframes for a ticker in the cache."""
    d = Path(cache_dir) if cache_dir else DATA_CACHE_DIR
    t = ticker.upper()
    found = []
    for canonical, ibkr_name in _IBKR_TIMEFRAME_MAP.items():
        if list(d.glob(f"{t}_{ibkr_name}_*.*")):
            found.append(canonical)
    return found
```

### 3.4 `features/pipeline.py` — Wire Local Intraday into Feature Pipeline

**File:** `features/pipeline.py`
**Lines to replace:** 700-745

Replace the entire intraday/LOB block with a version that falls back to local IBKR cache when WRDS is unavailable. The `compute_lob_features()` function at `lob_features.py:88` accepts any DataFrame with OHLCV columns and a DatetimeIndex — it does not require WRDS specifically. Only `compute_intraday_features()` from `intraday.py:22` is WRDS-specific (requires 1-minute TAQmsec bars).

Key logic:
1. Try WRDS TAQmsec first (keep existing behavior exactly as-is)
2. If WRDS unavailable, use `load_intraday_ohlcv(ticker, "30m")` from local cache
3. For each trading date in the last 60 days, slice that day's bars from the intraday DataFrame
4. Pass day-sliced bars to `compute_lob_features()`
5. Skip `compute_intraday_features()` for local data (it needs 1-min bars)
6. The ticker used for cache lookup should come from `df.attrs.get("ticker", str(permno))` since cache files are keyed by ticker symbol, not PERMNO

### 3.5 `config.py` — Add Intraday Configuration Constants

**File:** `config.py`
**Insert after line 104** (after `MIN_BARS = 500`):

```python
# ── Intraday Data ─────────────────────────────────────────────────────
INTRADAY_TIMEFRAMES = ["4h", "1h", "30m", "15m", "5m", "1m"]
INTRADAY_CACHE_SOURCE = "ibkr"
INTRADAY_MIN_BARS = 100
```

### 3.6 `data/local_cache.py` — Update `rehydrate_cache_metadata()` for Intraday + Parquet

**File:** `data/local_cache.py`
**Function:** `rehydrate_cache_metadata()` at line 325

The existing rehydrate function only processes daily files via `_daily_cache_files()`. It needs to:
1. Also process `*_daily_*.parquet` files (not just CSV)
2. Also process intraday files (`*_4hour_*`, `*_1hour_*`, `*_30min_*`)
3. For intraday files, skip the `_to_daily_ohlcv()` validation (which rejects sub-daily data)
4. Include `"timeframe"` in the metadata payload
5. Set source to `"wrds"` for daily files and `"ibkr"` for intraday files

### 3.7 `dash_ui/data/loaders.py` — Fix Benchmark Loading

**File:** `dash_ui/data/loaders.py`
**Function:** `load_benchmark_returns()`

Update to glob for `SPY_daily_*.parquet` and `SPY_daily_*.csv` as fallbacks after the exact `SPY_1d.parquet` / `84398_1d.parquet` paths fail.

### 3.8 `dash_ui/pages/data_explorer.py` — Add Timeframe Support

**File:** `dash_ui/pages/data_explorer.py`

1. Add a timeframe dropdown (Daily, 4H, 1H, 30M) to the controls row
2. Modify `_download_ticker()` at line 231 to try local cache first via `load_intraday_ohlcv()` for intraday or `load_ohlcv_with_meta()` for daily, falling back to yfinance only for daily
3. Chart rendering (candlestick, volume, SMA overlays) works unchanged — just ensure x-axis formats intraday timestamps correctly

### 3.9 `config.py` — Annotate Missing Tickers in UNIVERSE_FULL

Add a comment to `UNIVERSE_FULL` noting which 28 tickers are not yet in the cache, so you know what still needs to be downloaded.

---

## 4. Instructions for Claude Code (Terminal)

Copy the entire block below and paste it into Claude in your terminal. It contains all context needed to perform every code change without ambiguity.

---

```
I need you to integrate IBKR intraday data and fix WRDS daily data loading in my quant engine. Read every file I reference before making changes. Do NOT guess at code — read the actual source first.

## CURRENT DATA STATE

My cache is at quant_engine/data/cache/ with 910 files (455 parquet + 455 CSV duplicates).

File naming convention for ALL files (daily and intraday):
  {TICKER}_{timeword}_{start_date}_{end_date}.parquet
  {TICKER}_{timeword}_{start_date}_{end_date}.csv

Timeword values: "daily", "4hour", "1hour", "30min"
Example: AAPL_daily_2006-02-21_2024-12-31.parquet
Example: AAPL_4hour_2015-12-14_2026-02-20.parquet

Column format (all files, all timeframes): Date,Open,High,Low,Close,Volume
- Daily: Date is date-only (2006-02-21)
- Intraday: Date is timezone-aware (2015-12-14 09:30:00-05:00)
- Intraday has proper distinct OHLCV values
- Daily has O=H=L=C (explained below)

154 unique tickers. 155 daily files, 100 files each for 4hour/1hour/30min.
ZERO .meta.json files exist. The old PERMNO-keyed files (84398_1d.parquet etc.) are gone.

## DAILY O=H=L=C EXPLANATION

The daily data came from WRDS CRSP. The CRSP dsf table only has `prc` (closing price) — it does NOT have open/high/low as separate fields. The code at wrds_provider.py line 479 explicitly does:
  'Open': g['prc'],  # CRSP has Close only; set O=H=L=C
  'High': g['prc'],
  'Low':  g['prc'],
  'Close': g['prc'],

However, CRSP dsf DOES carry `openprc`, `askhi`, `bidlo` fields for many stocks. They just aren't being queried. This needs to be fixed in the SQL query.

## TASK 1: Fix WRDS CRSP query to include real OHLCV fields

Read data/wrds_provider.py — specifically the get_crsp_prices() function starting at line 406, the SQL query at line 445, and the DataFrame construction at line 478.

Modify the function:

1. Update the SQL query at line 446 to also SELECT a.openprc, a.askhi, a.bidlo:
   Currently: SELECT a.permno, a.date, a.prc, a.ret, a.vol, a.shrout, b.ticker, b.comnam
   Change to: SELECT a.permno, a.date, a.prc, a.openprc, a.askhi, a.bidlo, a.ret, a.vol, a.shrout, b.ticker, b.comnam

2. After the query returns, convert the new columns to numeric:
   df['openprc'] = pd.to_numeric(df.get('openprc'), errors='coerce')
   df['askhi'] = pd.to_numeric(df.get('askhi'), errors='coerce')
   df['bidlo'] = pd.to_numeric(df.get('bidlo'), errors='coerce')

3. Update the DataFrame construction at line 478-482:
   'Open':   g['openprc'].abs().fillna(g['prc'].abs()),
   'High':   g['askhi'].abs().fillna(g['prc'].abs()),
   'Low':    g['bidlo'].abs().fillna(g['prc'].abs()),
   'Close':  g['prc'].abs(),
   (Use .abs() because CRSP prices are negative when they represent bid/ask midpoints)

4. Update the docstring comment at line 429 — remove "Note: CRSP doesn't have Open/High/Low" and replace with "Note: Uses openprc/askhi/bidlo when available, falls back to prc (close) when not."

5. Also update get_crsp_prices_with_delistings() at line 496 — it calls self.get_crsp_prices() internally (line 510), so it inherits the fix. But verify this by reading the function.

## TASK 2: Fix daily parquet loading in data/local_cache.py

Read the FULL data/local_cache.py file first. The issues are:

Problem A: load_ohlcv_with_meta() at line 220 looks for {TICKER}_1d.parquet but files are named {TICKER}_daily_{dates}.parquet. There is no glob fallback for parquet.

Problem B: _daily_cache_files() at line 298-311 globs for *_daily_*.csv but NOT *_daily_*.parquet. This means the rehydrate function also misses parquet files.

Problem C: list_cached_tickers() at line 281-295 same issue — only globs *_daily_*.csv not parquet.

Problem D: _cache_meta_path() at line 121-125 only recognizes _1D. prefix, not _daily_.

Fix all four:

A. In load_ohlcv_with_meta(): After the exact parquet_path check (line 220-229), add a glob for root.glob(f"{ticker.upper()}_daily_*.parquet"). Try each match through _to_daily_ohlcv + _read_cache_meta. This should go BEFORE the CSV candidates loop.

B. In _daily_cache_files(): Add root.glob("*_daily_*.parquet") to the candidates list, between the *_1d.csv and *_daily_*.csv lines.

C. In list_cached_tickers(): Add a glob for *_daily_*.parquet with the same stem extraction logic.

D. In _cache_meta_path(): If "_DAILY_" or "_daily_" is in the filename, return data_path.with_suffix(".meta.json") (file-adjacent meta). Keep the existing _1D handling and the canonical fallback.

## TASK 3: Add intraday load function to data/local_cache.py

Add these new functions after load_ohlcv() (after line 279):

1. A module-level constant _IBKR_TIMEFRAME_MAP mapping canonical codes to IBKR filename words:
   {"4h": "4hour", "1h": "1hour", "30m": "30min", "15m": "15min", "5m": "5min", "1m": "1min"}

2. load_intraday_ohlcv(ticker, timeframe, cache_dir=None) -> Optional[pd.DataFrame]:
   - Maps timeframe to IBKR filename word using _IBKR_TIMEFRAME_MAP
   - Globs for {TICKER}_{ibkr_tf}_*.parquet first, then CSV fallback
   - Reads, normalizes columns with _normalize_ohlcv_columns(), validates REQUIRED_OHLCV present
   - Does NOT call _to_daily_ohlcv() (which rejects sub-daily data via the 12-hour check at line 84)
   - Sorts by index, removes NaN/duplicate indices, drops NaN OHLCV rows
   - Returns DataFrame or None

3. list_intraday_timeframes(ticker, cache_dir=None) -> List[str]:
   - Returns list of available canonical timeframe codes for a ticker
   - Checks existence via glob for each timeframe in _IBKR_TIMEFRAME_MAP

## TASK 4: Update rehydrate_cache_metadata for intraday + parquet

Read data/local_cache.py rehydrate_cache_metadata() at line 325 and run_rehydrate_cache_metadata.py.

The rehydrate function currently only processes files returned by _daily_cache_files(), which only finds daily CSV/parquet files. It needs to also process intraday files.

Modify rehydrate_cache_metadata() to:
1. Also scan intraday files: *_4hour_*.parquet, *_4hour_*.csv, *_1hour_*.parquet, *_1hour_*.csv, *_30min_*.parquet, *_30min_*.csv (and 15min, 5min, 1min for future)
2. For intraday files, skip _to_daily_ohlcv() validation — just validate OHLCV columns are present and read the file normally
3. Include "timeframe" in the metadata payload: "daily" for daily files, "4h"/"1h"/"30m" etc for intraday
4. Set source appropriately: use root_source_label mapping (which defaults to "wrds" for DATA_CACHE_DIR per line 349)
5. For the meta path of intraday files, use data_path.with_suffix(".meta.json") (file-adjacent)

Also update _daily_cache_files() since it was already fixed in Task 2 — or create a new _all_cache_files() helper that includes intraday patterns.

## TASK 5: Wire local intraday data into features/pipeline.py

Read features/pipeline.py lines 700-745, features/intraday.py (all 194 lines), and features/lob_features.py (first 150 lines, especially compute_lob_features at line 88).

The current code at pipeline.py lines 700-745 ONLY runs intraday/LOB features when WRDS TAQmsec is available. Modify this block to also try local IBKR cache.

Replace lines 700-745 with logic that:
1. Tries WRDS TAQmsec first (keep existing behavior EXACTLY)
2. If WRDS unavailable, falls back to local IBKR cache:
   - Import load_intraday_ohlcv from ..data.local_cache
   - Import compute_lob_features from .lob_features
   - For each permno/df in data:
     - Get ticker from df.attrs.get("ticker", str(permno))
     - Load 30min bars: load_intraday_ohlcv(ticker, "30m"). If None, try "1h"
     - If bars found and len(bars) > 100:
       - For each date in df.index[-60:] (last 60 trading days):
         - Slice that day's bars from the intraday DataFrame using bars.loc[date_str]
         - If the day has >= 5 bars, pass to compute_lob_features()
         - Append result with permno and date keys
   - Do NOT try compute_intraday_features() for local data — it needs 1-min WRDS TAQmsec bars
3. Join intraday_rows and lob_rows into features DataFrame same as existing code

IMPORTANT: The ticker for cache lookup must come from df.attrs.get("ticker", str(permno)) because cache files are keyed by ticker symbol (e.g., "AAPL"), NOT by PERMNO (e.g., "14593").

## TASK 6: Add intraday config constants to config.py

Read config.py first. Add after line 104 (after MIN_BARS = 500):

INTRADAY_TIMEFRAMES = ["4h", "1h", "30m", "15m", "5m", "1m"]
INTRADAY_CACHE_SOURCE = "ibkr"
INTRADAY_MIN_BARS = 100

Also add a comment to UNIVERSE_FULL (line 78-93) noting which 28 tickers are not yet in the cache:
AMD, INTC, CRM, ADBE, ORCL, DDOG, NET, CRWD, ZS, SNOW, MDB, PANW, FTNT, ABBV, TMO, MCD, TGT, COST, GS, BLK, MA, BA, CAVA, BROS, TOST, CHWY, ETSY, POOL

## TASK 7: Fix benchmark loading in dash_ui/data/loaders.py

Read dash_ui/data/loaders.py — specifically load_benchmark_returns().

The old SPY_1d.parquet and 84398_1d.parquet no longer exist. SPY is now at SPY_daily_2006-02-21_2024-12-31.parquet.

Update load_benchmark_returns() to:
1. Try exact paths first (SPY_1d.parquet, 84398_1d.parquet) for backward compatibility
2. Add glob fallback: DATA_CACHE_DIR.glob("SPY_daily_*.parquet"), then SPY_daily_*.csv
3. Use the first match found

## TASK 8: Add timeframe support to dash_ui/pages/data_explorer.py

Read the full data_explorer.py page first (750 lines).

Currently _download_ticker() at line 231 only uses yfinance. Modify to support intraday:

1. Add a timeframe dropdown to the controls row (after the Universe dropdown) with options:
   - "Daily" (value "daily")
   - "4 Hour" (value "4h")
   - "1 Hour" (value "1h")
   - "30 Min" (value "30m")

2. Modify _download_ticker() to accept a timeframe parameter:
   - If timeframe != "daily": use load_intraday_ohlcv(ticker, timeframe) from local cache. If cache miss, return None (no yfinance fallback for intraday).
   - If timeframe == "daily": try load_ohlcv_with_meta(ticker) from local cache first, then fall back to yfinance as current behavior.

3. Update the load_data callback to pass the selected timeframe to _download_ticker()

4. The candlestick chart, volume bars, and SMA overlays all consume OHLCV DataFrames identically regardless of timeframe — no chart changes needed.

## VERIFICATION

After all changes:
1. Syntax check: python3 -c "import ast; import glob; [ast.parse(open(f).read()) for f in glob.glob('quant_engine/**/*.py', recursive=True)]"
2. Run metadata rehydration: python -m quant_engine.run_rehydrate_cache_metadata --force
3. Verify daily loading: python3 -c "from quant_engine.data.local_cache import load_ohlcv_with_meta; df, meta, path = load_ohlcv_with_meta('AAPL'); print(f'AAPL daily: {len(df)} bars from {path}')"
4. Verify intraday loading: python3 -c "from quant_engine.data.local_cache import load_intraday_ohlcv, list_intraday_timeframes; print(list_intraday_timeframes('AAPL')); df = load_intraday_ohlcv('AAPL', '4h'); print(f'4h: {len(df)} bars'); df = load_intraday_ohlcv('AAPL', '30m'); print(f'30m: {len(df)} bars')"
5. Verify metadata exists: python3 -c "import glob; metas = glob.glob('quant_engine/data/cache/*.meta.json'); print(f'{len(metas)} metadata files created')"
6. Print summary: files modified, functions added, metadata files generated

## IMPORTANT RULES

- Do NOT delete or modify any cached data files (.parquet / .csv)
- Do NOT change _to_daily_ohlcv() — it correctly rejects intraday data for the daily pipeline
- Keep ALL existing code paths working (backward compat with _1d.parquet, _1d.csv naming)
- The intraday.py compute_intraday_features() requires 1-minute WRDS TAQmsec data — don't make it work with 30min IBKR bars. Only compute LOB features from local intraday data.
- Prefer parquet over CSV when both exist (parquet is faster and preserves dtypes)
- For the WRDS SQL fix, use .abs() on price fields because CRSP marks bid/ask midpoints as negative
```

---

## 5. Execution Order

The tasks should be executed in this order to avoid intermediate breakage:

| Step | Task | Why This Order |
|------|------|---------------|
| 1 | Fix `_cache_meta_path()` and parquet globs (Task 2) | Foundation — everything else depends on correct file finding |
| 2 | Add `load_intraday_ohlcv()` (Task 3) | Required by Tasks 4, 5, 8 |
| 3 | Add config constants (Task 6) | Quick, no dependencies |
| 4 | Update `rehydrate_cache_metadata()` (Task 4) | Depends on Tasks 2-3 for file finding |
| 5 | Run rehydrate script to generate all `.meta.json` files | Metadata must exist before loader trusted-source checks work |
| 6 | Fix WRDS CRSP query (Task 1) | Independent, can re-download data afterward |
| 7 | Wire pipeline intraday fallback (Task 5) | Depends on Task 3 |
| 8 | Fix benchmark loading (Task 7) | Independent |
| 9 | Add Data Explorer timeframe support (Task 8) | Depends on Task 3 |

---

## 6. Future Work (After This Integration)

### 6.1 Re-download Daily Data from WRDS

After fixing the CRSP query (Task 1), re-run the WRDS download for all tickers:
```bash
python -m quant_engine.run_train --download-only --tickers ALL --years 20
```
This refreshes daily cache files with proper OHLCV values and auto-generates `.meta.json` sidecars.

### 6.2 Download Missing 28 Tickers from IBKR

Add IBKR intraday downloads for: AMD, INTC, CRM, ADBE, ORCL, DDOG, NET, CRWD, ZS, SNOW, MDB, PANW, FTNT, ABBV, TMO, MCD, TGT, COST, GS, BLK, MA, BA, CAVA, BROS, TOST, CHWY, ETSY, POOL.

### 6.3 Download 1min/5min/15min Data from IBKR

Once available, place in cache with the same naming convention. The `load_intraday_ohlcv()` and `_IBKR_TIMEFRAME_MAP` already handle "15min", "5min", "1min" filenames.

With 1-minute data, `compute_intraday_features()` can be adapted to read from local cache, unlocking 6 additional microstructure features: intraday_vol_ratio, vwap_deviation, amihud_illiquidity, kyle_lambda, realized_vol_5m, microstructure_noise.

### 6.4 Update Daily Data to Feb 2026

Most daily files end at 2024-12-31. After fixing the WRDS query, re-download with end date of today to close the 14-month gap between daily and intraday coverage.

### 6.5 Create Multi-Timeframe Feature Module

The 4-hour bars (10 years of history) are a unique asset not available from WRDS. Consider a new `features/multi_timeframe.py`:
- 4h trend alignment vs daily trend
- 4h momentum divergence from daily momentum
- 4h realized volatility vs daily realized volatility
- Multi-timeframe RSI concordance
- Cross-timeframe volume profile

### 6.6 Clean Up CSV Duplicates

After confirming parquet loading works for all timeframes:
```bash
cd quant_engine/data/cache
rm *_daily_*.csv *_4hour_*.csv *_1hour_*.csv *_30min_*.csv
```
Saves ~200MB.

---

## 7. Summary

| Category | Count | Items |
|----------|-------|-------|
| Critical Issues | 3 | Daily O=H=L=C (WRDS query fix), file naming vs loader mismatch, no metadata files |
| High Issues | 3 | Old PERMNO files gone, intraday not wired to pipeline, daily ends Dec 2024 |
| Medium Issues | 2 | 28 missing tickers, duplicate CSV+parquet |
| Low Issues | 1 | 54 tickers daily-only |
| Code Files to Modify | 7 | wrds_provider.py, local_cache.py, pipeline.py, config.py, data_explorer.py, loaders.py (dash), run_rehydrate (verify) |
| New Functions to Add | 2 | load_intraday_ohlcv(), list_intraday_timeframes() |
| Tasks in Claude Code Prompt | 8 | WRDS fix, parquet loader, intraday loader, rehydrate update, pipeline wiring, config, benchmark fix, data explorer |
