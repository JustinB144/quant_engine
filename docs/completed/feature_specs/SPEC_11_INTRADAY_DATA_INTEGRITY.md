# Feature Spec: Intraday Data Integrity — Multi-Source Download with IBKR Truth Validation

> **Status:** Draft
> **Author:** Claude (Anthropic) for Justin
> **Date:** 2026-02-26
> **Estimated effort:** 45 hours across 7 tasks

---

## Why

The quant engine is expanding to intraday timeframes (1m, 5m, 15m, 30m, 1h, 4h) with 10 years of history across 128 tickers. Free data sources (Alpaca, Alpha Vantage) provide the volume but **their data quality is unverified and potentially compromised** — missing bars, stale prices, incorrect splits, phantom volume, timezone errors, and off-exchange prints can silently corrupt feature computation, regime detection, and backtest results. IBKR data (exchange-sourced, SIP-aggregated) is the most trustworthy retail source but has severe rate limits (~6 req/min). The system must use free sources for bulk download while using IBKR as the ground-truth validation layer — and **reject or replace any data that doesn't pass**.

Corrupted intraday data is catastrophic for a quant system because:
- Features computed on bad bars propagate silently through the entire pipeline
- Regime detection on stale/phantom data produces false state transitions
- Backtest PnL on incorrect close prices gives meaningless Sharpe ratios
- Position sizing calibrated on bad volatility estimates produces wrong allocations

## What

A multi-source intraday download system with:
1. **Primary download** from Alpaca (free, 200 req/min, 10+ years depth)
2. **Secondary download** from Alpha Vantage (20+ year depth, month-by-month, paid tier for speed)
3. **IBKR truth validation** — systematic cross-source comparison on stratified date samples
4. **13-point per-bar quality gate** — every bar must pass before entering the cache
5. **Automatic IBKR replacement** — bars that fail validation are replaced, not patched
6. **Per-ticker quality reports** persisted as JSON sidecars for audit trail
7. **Hard rejection** — tickers that fail >5% of bars across the validated sample are quarantined

## Constraints

### Must-haves
- IBKR is the truth source. If Alpaca and IBKR disagree, IBKR wins. Always.
- Every downloaded file gets a `.quality.json` sidecar documenting all checks performed
- Close price tolerance between Alpaca and IBKR must be ≤0.15% for normal bars, ≤1.0% for split-adjacent bars
- OHLC relationship enforcement: Low ≤ Open ≤ High, Low ≤ Close ≤ High (always)
- No bar enters the cache without passing the quality gate
- Validation must sample across the FULL date range (not just recent days)
- Split/dividend detection must compare against known corporate action dates
- All timezone handling must be explicit (ET for US equities, UTC internally)

### Must-nots
- Must NOT silently accept data that fails validation — must either replace or quarantine
- Must NOT assume Alpaca data is correct because "it's close enough"
- Must NOT validate only recent data — must sample old data too (2016-era bars are the most likely to be wrong)
- Must NOT use `bare except` — every error path must be logged with the specific failure reason

### Out of scope
- Real-time streaming validation (this is for historical bulk download only)
- Tick-level or sub-minute data
- Non-US equity markets

## Current State

### Key files
| File | Role | Notes |
|------|------|-------|
| `data/quality.py` | Daily OHLCV quality checks | Has `assess_ohlcv_quality()` with 5 checks; daily-only, no intraday awareness |
| `data/local_cache.py` | Cache read/write with normalize + meta | `_normalize_ohlcv_columns()`, `_write_cache_meta()` — reuse these |
| `scripts/ibkr_intraday_download.py` | IBKR chunked downloader | `download_intraday_chunked()` with `--pace` flag, 2s default |
| `scripts/alpaca_intraday_download.py` | Hybrid downloader (current, shallow) | Has placeholder quality_check() — needs complete rewrite |
| `config.py` | Central config | `DATA_CACHE_DIR`, `UNIVERSE_INTRADAY` (128 tickers), `INTRADAY_TIMEFRAMES` |

### Existing patterns to follow
- Quality reports use `DataQualityReport` dataclass (data/quality.py)
- Cache files: `{TICKER}_{timeword}_{start}_{end}.parquet` with `.meta.json` sidecars
- Config thresholds: `MAX_MISSING_BAR_FRACTION=0.05`, `MAX_ZERO_VOLUME_FRACTION=0.25`, `MAX_ABS_DAILY_RETURN=0.40`
- NYSE calendar via `pandas_market_calendars` with `pd.bdate_range` fallback
- Logging via `logging.getLogger(__name__)`

### Configuration
- `INTRADAY_TIMEFRAMES = ["4h", "1h", "30m", "15m", "5m", "1m"]`
- `UNIVERSE_INTRADAY` = 128 tickers
- `MARKET_OPEN = "09:30"`, `MARKET_CLOSE = "16:00"` (ET)

## Tasks

### T1: Intraday Quality Gate — 13-Point Per-Bar Validation

**What:** Create `data/intraday_quality.py` with a comprehensive bar-level and series-level quality gate that catches every known class of intraday data corruption.

**Files:**
- `data/intraday_quality.py` — new file, all quality logic
- `config.py` — add intraday quality thresholds

**Implementation notes:**

The 13 checks, in order:

**Per-bar structural checks (instant rejection):**
1. **OHLC consistency**: `High >= max(Open, Close)` AND `Low <= min(Open, Close)` AND `High >= Low`. Any violation = bar rejected.
2. **Non-negative volume**: `Volume >= 0`. Negative volume = bar rejected.
3. **Non-negative prices**: All of O, H, L, C must be > 0. Zero or negative price = bar rejected.
4. **Timestamp in RTH**: Bar timestamp must fall within 09:30-16:00 ET on a NYSE trading day. Bars outside RTH = rejected (we requested RTH-only data).

**Per-bar statistical checks (flag + investigate):**
5. **Extreme bar return**: `|Close/prev_Close - 1| > 0.15` for 1m bars (scale threshold by timeframe: 1m=15%, 5m=20%, 15m=25%, 30m=30%, 1h=35%, 4h=50%). Flag but don't auto-reject — could be legitimate gap.
6. **Stale price detection**: If Close == prev_Close AND High == Low == Open == Close for 3+ consecutive bars, the data is stale (frozen feed). Flag entire stale run.
7. **Zero volume on liquid name**: If ticker is in top-50 by market cap and volume == 0 during regular hours, flag as suspicious.

**Series-level checks (after all bars loaded):**
8. **Missing bar ratio**: Count expected bars (using NYSE calendar + timeframe) vs actual bars. Fail if missing > 5%.
9. **Duplicate timestamps**: Any duplicate timestamps = fail (after dedup, check if dedup changed bar count by >0.1%).
10. **Monotonic index**: Timestamps must be strictly increasing after sort. Non-monotonic = data corruption.
11. **Overnight gap sanity**: First bar of each day's Open should be within 10% of previous day's last Close (for non-split days). Larger gaps flagged for split check.
12. **Volume distribution sanity**: Median daily volume must be > 100 shares for any ticker in UNIVERSE_INTRADAY. If median daily volume < 100, quarantine.
13. **Split detection**: If any bar-to-bar return exceeds 40%, check if date is near a known split date (can be detected by exact 2:1, 3:1, 4:1, 5:1, 10:1, 3:2, 20:1 ratio). If ratio matches split pattern but data isn't adjusted, flag as unadjusted-split corruption.

Return structure:
```python
@dataclass
class IntradayQualityReport:
    ticker: str
    timeframe: str
    source: str
    passed: bool                    # True only if ALL hard checks pass
    total_bars: int
    rejected_bars: int              # Bars that failed hard checks
    flagged_bars: int               # Bars that failed soft checks
    checks: Dict[str, CheckResult]  # Per-check detail
    quality_score: float            # 0.0 - 1.0 composite
    quarantine: bool                # True if ticker should be quarantined

@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str                   # "hard" or "soft"
    count: int                      # Number of bars affected
    details: str                    # Human-readable explanation
```

**Verify:**
```bash
python -c "
from data.intraday_quality import validate_intraday_bars, IntradayQualityReport
import pandas as pd, numpy as np
# Create synthetic bad data
idx = pd.date_range('2024-01-02 09:30', periods=100, freq='1min')
df = pd.DataFrame({
    'Open': np.random.uniform(100, 101, 100),
    'High': np.random.uniform(99, 100, 100),   # H < O — should fail check 1
    'Low': np.random.uniform(100, 101, 100),    # L > H — should fail check 1
    'Close': np.random.uniform(100, 101, 100),
    'Volume': np.random.randint(0, 1000, 100),
}, index=idx)
report = validate_intraday_bars(df, 'TEST', '1m')
assert not report.passed, 'Should fail on OHLC inconsistency'
assert report.checks['ohlc_consistency'].count > 0
print(f'PASS: {report.rejected_bars} bars rejected, quality={report.quality_score:.3f}')
"
```

---

### T2: IBKR Cross-Source Validator — Stratified Date Sampling

**What:** Create `data/cross_source_validator.py` that systematically validates Alpaca/AV data against IBKR using stratified sampling across the full date range, not just recent days.

**Files:**
- `data/cross_source_validator.py` — new file
- `config.py` — add cross-validation thresholds

**Implementation notes:**

The current `validate_with_ibkr()` in the downloader is dangerously shallow — it only checks 5 recent days and uses a 1.0% tolerance. Problems with this approach:
- Recent data is the LEAST likely to be wrong (APIs prioritize fresh data)
- 1.0% tolerance on Close is huge — a $100 stock could be off by $1 and pass
- Only checks Close, ignoring H/L/O/V which could be completely wrong
- Doesn't check for missing bars, only price differences on overlapping timestamps
- Swallows all exceptions silently

The replacement must:

**Stratified sampling strategy:**
- Divide the full date range into 10 equal windows
- From each window, select 2 random trading days
- Download those 20 days from IBKR (at 2s pace = ~40s per ticker, acceptable)
- Compare every overlapping bar across ALL OHLCV columns

**Comparison logic per bar:**
- Close tolerance: ≤0.15% (was 1.0% — tighten by 6.7x)
- Open tolerance: ≤0.20% (opens can vary slightly by exchange)
- High tolerance: ≤0.25%
- Low tolerance: ≤0.25%
- Volume tolerance: ≤5.0% OR ≤100 shares absolute difference (volume varies by SIP vs single exchange)
- Timestamp alignment: Alpaca bar timestamp must match IBKR within the same minute

**Failure modes:**
- `PRICE_MISMATCH`: Close differs by >0.15%. Action: **replace bar with IBKR data**.
- `MISSING_IN_ALPACA`: Bar exists in IBKR but not Alpaca. Action: **insert IBKR bar**.
- `PHANTOM_IN_ALPACA`: Bar exists in Alpaca but not IBKR. Action: **flag for review** (could be extended hours leaking in).
- `SPLIT_MISMATCH`: Price ratio suggests unadjusted split in one source. Action: **quarantine ticker, download fresh from IBKR**.
- `VOLUME_ANOMALY`: Volume differs by >50%. Action: **log warning, keep Alpaca** (volume aggregation differs by source).

**Return structure:**
```python
@dataclass
class CrossValidationReport:
    ticker: str
    timeframe: str
    sample_windows: int             # How many date windows sampled
    sample_days: int                # Total days sampled
    overlapping_bars: int           # Bars compared
    price_mismatches: int           # Bars where close > 0.15% off
    missing_in_primary: int         # Bars in IBKR but not Alpaca
    phantom_in_primary: int         # Bars in Alpaca but not IBKR
    split_mismatches: int           # Suspected unadjusted split bars
    volume_anomalies: int           # Volume > 50% different
    bars_replaced: int              # Bars swapped to IBKR data
    bars_inserted: int              # IBKR bars added to fill gaps
    passed: bool                    # True if mismatch rate < 5%
    mismatch_rate: float            # price_mismatches / overlapping_bars
    details: List[dict]             # Per-bar mismatch details for audit
```

**Verify:**
```bash
python -c "
from data.cross_source_validator import CrossSourceValidator
# Verify class loads and has correct methods
v = CrossSourceValidator.__new__(CrossSourceValidator)
assert hasattr(v, 'validate_ticker')
assert hasattr(v, 'replace_bad_bars')
print('PASS: CrossSourceValidator structure verified')
"
```

---

### T3: Alpha Vantage Provider — Month-by-Month Historical Fetch

**What:** Create `data/providers/alpha_vantage_provider.py` that fetches intraday data from Alpha Vantage using the `month` parameter for historical depth, with proper rate limiting and error handling.

**Files:**
- `data/providers/alpha_vantage_provider.py` — new file
- `data/providers/__init__.py` — new file (package init)

**Implementation notes:**

Alpha Vantage provides 20+ years of intraday history using the `month=YYYY-MM` parameter, one month per request. For 10 years × 12 months = 120 requests per ticker per timeframe.

Key implementation details:
- Use `alpha_vantage` PyPI package for structured API calls
- Free tier: 25 requests/day (useless). Paid $49.99/mo: 75 req/min (viable for secondary source).
- Must handle: HTTP 429 (rate limit) with exponential backoff, empty responses (weekends/holidays), API key rotation
- Convert AV column names (`1. open`, `2. high`, etc.) to canonical OHLCV
- AV returns ET timestamps — normalize to tz-naive ET (matching our convention)
- AV `extended_hours=false` to match IBKR RTH data

Provider interface:
```python
class AlphaVantageProvider:
    def __init__(self, api_key: str, pace: float = 0.85):
        """pace=0.85s = ~70 req/min, safe for $49.99 tier (75/min limit)"""

    def fetch_month(self, ticker: str, timeframe: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Fetch one month of intraday data."""

    def fetch_range(self, ticker: str, timeframe: str, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
        """Fetch a date range by iterating months. Concatenates and deduplicates."""
```

**Verify:**
```bash
python -c "
from data.providers.alpha_vantage_provider import AlphaVantageProvider
# Verify structure without API key
p = AlphaVantageProvider.__new__(AlphaVantageProvider)
assert hasattr(p, 'fetch_month')
assert hasattr(p, 'fetch_range')
print('PASS: AlphaVantageProvider structure verified')
"
```

---

### T4: Rebuild Alpaca Provider with Proper Error Handling

**What:** Refactor the Alpaca download logic from the script into a proper provider class in `data/providers/alpaca_provider.py` with robust error handling, timezone management, and rate limit recovery.

**Files:**
- `data/providers/alpaca_provider.py` — new file (extract from alpaca_intraday_download.py)
- `data/providers/__init__.py` — update exports

**Implementation notes:**

Extract `download_alpaca_chunked()` from the script into a class:
```python
class AlpacaProvider:
    def __init__(self, api_key: str, api_secret: str, pace: float = 0.35):
        ...

    def fetch_range(self, ticker: str, timeframe: str, target_days: int) -> Optional[pd.DataFrame]:
        """Download with chunking, rate limit recovery, and proper tz handling."""

    def _handle_rate_limit(self, retry_count: int) -> float:
        """Exponential backoff: 10s, 30s, 60s, 120s. Max 4 retries."""
```

Key fixes over current implementation:
- Explicit timezone handling: Alpaca returns UTC → convert to ET → strip tz (matching IBKR convention)
- Rate limit detection: Check for HTTP 429, `too many requests`, and `rate limit` in error messages
- Exponential backoff instead of flat 30s sleep
- Log every failed request with ticker, timeframe, date range, error message
- Return `None` only after all retries exhausted — never silently drop data

**Verify:**
```bash
python -c "
from data.providers.alpaca_provider import AlpacaProvider
p = AlpacaProvider.__new__(AlpacaProvider)
assert hasattr(p, 'fetch_range')
assert hasattr(p, '_handle_rate_limit')
print('PASS: AlpacaProvider structure verified')
"
```

---

### T5: Rebuild Hybrid Downloader with Full Validation Pipeline

**What:** Rewrite `scripts/alpaca_intraday_download.py` to use the provider classes and validation pipeline, implementing the full download → quality gate → cross-validate → replace → save workflow.

**Files:**
- `scripts/alpaca_intraday_download.py` — full rewrite using provider classes
- `config.py` — add intraday validation config block

**Implementation notes:**

The download pipeline for each ticker/timeframe:

```
1. DOWNLOAD from Alpaca (primary)
   ├── If Alpaca fails → try Alpha Vantage (secondary)
   ├── If both fail → try IBKR (tertiary, slow but reliable)
   └── If all fail → skip ticker, log failure

2. QUALITY GATE (13-point check from T1)
   ├── Reject bars that fail hard checks (OHLC, negative price/volume, outside RTH)
   ├── Flag bars that fail soft checks (extreme returns, stale prices)
   └── If rejected_bars > 5% → quarantine ticker

3. IBKR CROSS-VALIDATION (stratified sampling from T2)
   ├── Sample 20 days across full date range
   ├── Compare every OHLCV column at 0.15% close tolerance
   ├── Replace mismatched bars with IBKR data
   ├── Insert missing bars from IBKR
   └── If mismatch_rate > 5% → quarantine ticker, re-download entirely from IBKR

4. SECOND QUALITY GATE (re-run after replacements)
   └── Ensure IBKR replacement bars also pass all checks

5. SAVE with full audit trail
   ├── .parquet data file
   ├── .meta.json (source, date range, bar count)
   └── .quality.json (full IntradayQualityReport + CrossValidationReport)
```

New config block:
```python
# ── Intraday Data Integrity ────────────────────────────────────────
INTRADAY_CLOSE_TOLERANCE_PCT = 0.15          # Max acceptable close price diff vs IBKR
INTRADAY_OPEN_TOLERANCE_PCT = 0.20
INTRADAY_HIGHLOW_TOLERANCE_PCT = 0.25
INTRADAY_VOLUME_TOLERANCE_PCT = 5.0
INTRADAY_MAX_REJECTED_BAR_PCT = 5.0          # Quarantine if >5% bars rejected
INTRADAY_MAX_MISMATCH_RATE_PCT = 5.0         # Quarantine if >5% bars mismatch vs IBKR
INTRADAY_VALIDATION_SAMPLE_WINDOWS = 10      # Number of date windows for stratified sampling
INTRADAY_VALIDATION_DAYS_PER_WINDOW = 2      # Days sampled per window
INTRADAY_QUARANTINE_DIR = DATA_CACHE_DIR / "quarantine"  # Quarantined data goes here
```

New CLI flags:
```
--source alpaca|alphavantage|ibkr    Primary source (default: alpaca)
--fallback-source alphavantage|ibkr  Secondary source (default: ibkr)
--skip-validation                     Skip IBKR cross-validation (NOT recommended)
--quarantine-report                   Print quarantine summary and exit
--revalidate                          Re-run validation on existing cache files
--tolerance FLOAT                     Override close price tolerance (default: 0.15%)
```

**Verify:**
```bash
python scripts/alpaca_intraday_download.py --dry-run --timeframes 1h --tickers AAPL
# Should print survey without downloading, showing validation config
```

---

### T6: Quality Report JSON Sidecar and Quarantine System

**What:** Implement the `.quality.json` sidecar format and the quarantine directory system so every cached file has a verifiable audit trail and bad data is isolated.

**Files:**
- `data/intraday_quality.py` — add `write_quality_report()`, `read_quality_report()`, `quarantine_ticker()`
- `data/local_cache.py` — update `_write_cache_meta()` to include quality hash

**Implementation notes:**

`.quality.json` sidecar format:
```json
{
  "ticker": "AAPL",
  "timeframe": "1m",
  "source": "alpaca+ibkr",
  "validated_at": "2026-02-26T14:30:00",
  "quality_gate": {
    "passed": true,
    "total_bars": 487520,
    "rejected_bars": 12,
    "flagged_bars": 45,
    "quality_score": 0.9847,
    "checks": {
      "ohlc_consistency": {"passed": true, "count": 0, "severity": "hard"},
      "non_negative_volume": {"passed": true, "count": 0, "severity": "hard"},
      "non_negative_prices": {"passed": true, "count": 0, "severity": "hard"},
      "timestamp_in_rth": {"passed": true, "count": 3, "severity": "hard"},
      "extreme_bar_return": {"passed": true, "count": 8, "severity": "soft"},
      "stale_price": {"passed": true, "count": 2, "severity": "soft"},
      "zero_volume_liquid": {"passed": true, "count": 12, "severity": "soft"},
      "missing_bar_ratio": {"passed": true, "count": 0, "severity": "hard"},
      "duplicate_timestamps": {"passed": true, "count": 0, "severity": "hard"},
      "monotonic_index": {"passed": true, "count": 0, "severity": "hard"},
      "overnight_gap": {"passed": true, "count": 5, "severity": "soft"},
      "volume_distribution": {"passed": true, "count": 0, "severity": "hard"},
      "split_detection": {"passed": true, "count": 0, "severity": "hard"}
    }
  },
  "cross_validation": {
    "passed": true,
    "sample_windows": 10,
    "sample_days": 20,
    "overlapping_bars": 7800,
    "price_mismatches": 3,
    "bars_replaced": 3,
    "bars_inserted": 12,
    "mismatch_rate": 0.0004
  },
  "data_hash": "sha256:abcdef1234..."
}
```

Quarantine system:
- Quarantined files go to `data/cache/quarantine/{TICKER}_{timeword}_{date}.parquet`
- Quarantine log at `data/cache/quarantine/quarantine_log.json` — append-only
- `quarantine_ticker()` moves file + meta + quality to quarantine dir
- `list_quarantined()` returns current quarantine inventory
- Re-running the downloader on a quarantined ticker downloads fresh from IBKR only

**Verify:**
```bash
python -c "
from data.intraday_quality import write_quality_report, read_quality_report
from pathlib import Path
import tempfile, json
# Write and read back
with tempfile.NamedTemporaryFile(suffix='.quality.json', delete=False) as f:
    path = Path(f.name)
write_quality_report(path, {'ticker': 'TEST', 'passed': True})
report = read_quality_report(path)
assert report['ticker'] == 'TEST'
print('PASS: Quality sidecar read/write works')
"
```

---

### T7: Integration Tests — End-to-End with Synthetic Data

**What:** Create comprehensive tests that verify the full pipeline works correctly using synthetic data with known corruptions, without requiring live API connections.

**Files:**
- `tests/test_intraday_quality.py` — unit tests for all 13 checks
- `tests/test_cross_source_validator.py` — tests for cross-validation logic

**Implementation notes:**

Test cases for the quality gate:
1. **Clean data** → all checks pass, quality_score ≈ 1.0
2. **OHLC violation** (High < Low) → hard reject, specific bars identified
3. **Negative volume** → hard reject
4. **Zero price** → hard reject
5. **Outside RTH** (Saturday timestamp) → hard reject
6. **Extreme return** (50% in 1 minute) → soft flag
7. **Stale prices** (5 identical bars) → soft flag
8. **Missing 10% of bars** → hard fail on series check
9. **Duplicate timestamps** → hard fail
10. **Known split** (exact 2:1 ratio) → split detection fires
11. **Mixed corruptions** → correct count of each type

Test cases for cross-validation:
1. **Identical data** → pass, 0 mismatches
2. **0.10% close difference** → pass (under 0.15% tolerance)
3. **0.20% close difference** → fail, bar replaced
4. **Missing bars in primary** → detected, counted
5. **2:1 price ratio** (split mismatch) → quarantine triggered
6. **All bars wrong** → quarantine triggered (>5% mismatch rate)

All tests use synthetic DataFrames — no API calls.

**Verify:**
```bash
python -m pytest tests/test_intraday_quality.py tests/test_cross_source_validator.py -v
```

---

## Validation

### Acceptance criteria
1. Every cached intraday parquet file has a `.quality.json` sidecar documenting all 13 checks
2. No bar in the cache violates OHLC consistency (High >= Low, etc.)
3. Cross-validated close prices match IBKR within 0.15% on ≥95% of sampled bars
4. Tickers with >5% mismatch rate are automatically quarantined (moved to quarantine dir)
5. IBKR replacement bars themselves pass the full 13-point quality gate
6. Split detection catches bars with exact 2:1, 3:1, 4:1, 5:1, 10:1 ratios
7. Missing bar ratio is computed against NYSE calendar (not naive business days)
8. Quality gate correctly rejects bars outside 09:30-16:00 ET
9. Stale price detection catches ≥3 consecutive identical OHLC bars
10. Quarantine log is append-only and human-readable
11. Full test suite passes with 100% of synthetic corruption cases caught
12. Downloader can recover from rate limits without losing data or crashing
13. Alpha Vantage month-by-month fetch correctly concatenates across year boundaries

### Verification steps
```bash
# Run unit tests
python -m pytest tests/test_intraday_quality.py tests/test_cross_source_validator.py -v

# Dry run to verify config
python scripts/alpaca_intraday_download.py --dry-run --tickers AAPL MSFT --timeframes 1h

# Single-ticker test with validation
python scripts/alpaca_intraday_download.py --tickers AAPL --timeframes 1h --years 1 --validate-ibkr

# Check quality sidecar was created
cat data/cache/AAPL_1hour_*.quality.json | python -m json.tool

# Check no quarantined tickers (for known liquid names)
python scripts/alpaca_intraday_download.py --quarantine-report
```

### Rollback plan
- All new code is in new files (`data/intraday_quality.py`, `data/cross_source_validator.py`, `data/providers/*.py`)
- Existing `data/quality.py` and `data/local_cache.py` are NOT modified (only extended)
- Original `scripts/ibkr_intraday_download.py` remains functional and unmodified
- Config additions use new constants — no existing constants changed
- To disable: set `INTRADAY_VALIDATION_ENABLED = False` in config.py (will be added as a feature flag)

---

## Notes

### Alpha Vantage Assessment
- **Viable as secondary source** at $49.99/mo tier (75 req/min) or higher
- Free tier (25 req/day) is useless for 128 tickers
- Provides 20+ years of intraday history via `month=YYYY-MM` parameter — deeper than Alpaca
- SIP-aggregated data (same quality tier as Alpaca)
- 120 API calls per ticker per timeframe for 10 years (12 months × 10 years)
- At 75 req/min: ~1.6 min per ticker, ~3.4 hours for 128 tickers per timeframe
- Main value: fills gaps where Alpaca data doesn't go back far enough

### Data Source Priority Order
1. **Alpaca** (primary) — free, 200 req/min, 10+ year depth, all timeframes
2. **Alpha Vantage** (secondary) — paid, 75 req/min, 20+ year depth, fills Alpaca gaps
3. **IBKR** (truth/validation/tertiary) — rate-limited but exchange-sourced, always wins disputes

### Why 0.15% Close Tolerance
- A $150 stock (typical large cap) at 0.15% tolerance = $0.225 max difference
- This catches: wrong exchange source, missing split adjustment, stale quotes, off-exchange prints
- This allows: normal SIP aggregation differences, sub-penny rounding, exchange vs consolidated close
- 1.0% tolerance (the old value) on a $150 stock = $1.50 — could hide a partially-applied split

### Why Stratified Sampling (Not Random)
- Old data (2016-2019) is most likely to have split/adjustment errors
- Middle data (2019-2022) may have COVID-era data gaps
- Recent data (2023-2026) is least likely to be wrong but still needs verification
- Stratified sampling guarantees coverage across all epochs
- 20 days × ~390 bars/day (1m) = ~7,800 bars validated per ticker — statistically meaningful

### Corporate Action Detection
- Known split ratios to check: 2:1, 3:1, 4:1, 5:1, 10:1, 3:2, 20:1
- If bar-to-bar return is within 1% of a known split ratio, check the date
- Reverse splits (1:N) also detected (return near -50%, -67%, -75%, etc.)
- This catches the most common data corruption: split applied in one source but not the other
