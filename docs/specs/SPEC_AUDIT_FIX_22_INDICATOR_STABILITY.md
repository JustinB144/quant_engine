# SPEC_AUDIT_FIX_22: Indicator Numerical Stability Fixes

**Priority:** HIGH — TailRiskAnalyzer produces ~1e9 values that survive winsorization; 30+ indicator classes have unguarded divisions producing silent inf→NaN conversions.
**Scope:** `indicators/tail_risk.py`, `indicators/indicators.py`
**Estimated effort:** 4 hours
**Depends on:** Nothing
**Blocks:** Nothing

---

## Context

The indicator subsystem has two numerical stability issues. First (F-03/P1), `TailRiskAnalyzer.semi_relative_modulus` at tail_risk.py:180-184 uses a fallback branch that divides by `1e-10` when upside variance is near zero, producing finite values up to ~1e9 during sustained drawdowns — these survive the pipeline's `inf→NaN` conversion and expanding-window winsorization because they are finite, dominating tree splits or gradient updates. Second (F-04/P2), 30+ indicator classes in indicators.py have divisions where the denominator can be zero (flat bars, zero volume periods, constant volatility) with no explicit guard. The pipeline converts resulting `inf` to `NaN` at line 1132, but this is silent information loss — the feature value is replaced by median fill in the predictor without any warning. The 5 structural analyzers (Spectral, SSA, TailRisk core, OT, Eigenvalue) demonstrate best practices with excellent guards (`max(..., 1e-10)`, `if S > 1e-12`, etc.) that should be applied to the main indicator classes.

---

## Tasks

### T1: Cap TailRiskAnalyzer semi_relative_modulus Fallback

**Problem:** `tail_risk.py:180-184`:
```python
if up_var > 1e-15:
    srm[i] = np.sqrt(down_var / up_var)
else:
    srm[i] = np.sqrt(down_var) / 1e-10 if down_var > 0 else 0.0
```
When `up_var ≤ 1e-15` and `down_var = 0.01`, the result is `sqrt(0.01) / 1e-10 ≈ 1e9`. These finite extreme values survive both `inf→NaN` conversion and winsorization (if multiple occur, the 99th percentile itself is inflated).

**File:** `indicators/tail_risk.py`

**Implementation:**
1. Replace the fallback with a capped asymmetry measure:
   ```python
   _SRM_CAP = 10.0  # Maximum semi-relative modulus value

   if up_var > 1e-15:
       srm[i] = min(np.sqrt(down_var / up_var), _SRM_CAP)
   else:
       # No upside variance — extreme downside dominance.
       # Cap to _SRM_CAP to prevent magnitude explosion.
       srm[i] = _SRM_CAP if down_var > 0 else 0.0
   ```
2. The cap value of 10.0 means "downside variance is 100x upside variance" — already an extreme reading that signals severe downside dominance. Values beyond this are noise, not signal.
3. Also cap the normal branch (`up_var > 1e-15`) because `up_var = 1e-14` and `down_var = 0.01` still produces `sqrt(0.01 / 1e-14) ≈ 1e6`.
4. Add a unit test:
   ```python
   def test_semi_relative_modulus_capped():
       """Verify SRM doesn't exceed cap even during extreme drawdown."""
       analyzer = TailRiskAnalyzer(window=20)
       # All negative returns (zero upside variance)
       returns = np.full(100, -0.01)
       srm = analyzer.compute_semi_relative_modulus(returns)
       assert np.all(srm[20:] <= 10.0), f"SRM exceeded cap: max={srm.max()}"
   ```

**Acceptance:** `SemiRelMod_20` never exceeds 10.0 in absolute value. A test with all-negative returns confirms the cap.

---

### T2: Add Division Guards to Unguarded Indicator Classes

**Problem:** 30+ indicator classes divide by values that can be zero (flat bars where `High == Low`, zero-volume periods, constant-volatility windows). The pipeline catches resulting `inf` at line 1132, but this is silent and loses information.

**File:** `indicators/indicators.py`

**Implementation:** Apply the `.replace(0, np.nan)` pattern already used by guarded indicators (ZScore:2161, VarianceRatio:2183, AmihudIlliquidity:2363). For each indicator, the fix is mechanical — add a guard to the denominator. Here are the specific fixes grouped by trigger condition:

**Group A: Flat-bar divisions (`High - Low` or `high_max - low_min`):**

```python
# Stochastic (line 258)
range_ = high_max - low_min
range_ = range_.replace(0, np.nan)  # Guard flat bars
k = 100 * (df['Close'] - low_min) / range_

# WilliamsR (line 300) — same pattern
range_ = high_max - low_min
range_ = range_.replace(0, np.nan)

# CandleBody (line 615)
range_ = df['High'] - df['Low']
range_ = range_.replace(0, np.nan)

# AccumulationDistribution (line 900)
range_ = df['High'] - df['Low']
range_ = range_.replace(0, np.nan)
```

**Group B: Zero-volume divisions:**

```python
# VolumeRatio (line 490)
avg_vol = df['Volume'].rolling(window=self.period).mean()
avg_vol = avg_vol.replace(0, np.nan)

# RVOL (line 826) — same pattern
avg_vol = avg_vol.replace(0, np.nan)

# VWAP (line 1332)
cum_vol = (df['Volume']).rolling(window=self.period).sum()
cum_vol = cum_vol.replace(0, np.nan)

# MFI (line 554)
negative_mf_sum = negative_mf.rolling(window=self.period).sum()
negative_mf_sum = negative_mf_sum.replace(0, np.nan)
```

**Group C: ATR/volatility divisions:**

```python
# ADX (line 435) — multiple unguarded divisions
atr = atr.replace(0, np.nan)  # Guard before plus_di/minus_di computation
di_sum = plus_di + minus_di
di_sum = di_sum.replace(0, np.nan)  # Guard DX computation

# NATR — same atr guard
# VolatilityRegime (line 1302)
vol_std = vol_std.replace(0, np.nan)

# NetVolumeTrend, TrendStrength — similar pattern
```

**Group D: All remaining indicators from the audit list:**
Apply the same `.replace(0, np.nan)` pattern to: CCI, PriceVsSMA, SMASlope, GapPercent, DistanceFromHigh, DistanceFromLow, ATRTrailingStop, ATRChannel, RiskPerATR, PriceVsVWAP, VWAPBands, PriceVsPOC, ValueAreaPosition, ShannonEntropy, FractalDimension, DominantCycle.

**Each fix follows this template:**
```python
# BEFORE (unguarded):
result = numerator / denominator

# AFTER (guarded):
denominator = denominator.replace(0, np.nan)  # or: where(abs(denominator) > 1e-12, denominator, np.nan)
result = numerator / denominator
```

**Acceptance:** None of the 30 indicator classes produce `inf` values for flat-bar, zero-volume, or zero-volatility input. A test with a flat-bar DataFrame (all OHLC equal) returns NaN (not inf) for affected indicators.

---

### T3: Add np.log() Positivity Guards

**Problem:** 5 indicator classes use `np.log()` on price ratios that are positive in practice but lack explicit positivity guards: ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility, HurstExponent, DFA.

**File:** `indicators/indicators.py`

**Implementation:**
1. For each `np.log(x)` call on price ratios, add a positivity guard:
   ```python
   # Pattern for price ratio logs:
   ratio = df['High'] / df['Low']
   ratio = ratio.clip(lower=1e-10)  # Guard against zero/negative ratios
   log_ratio = np.log(ratio)
   ```
2. Apply to all 5 classes. The `clip(lower=1e-10)` approach is preferable to `.replace(0, np.nan)` because log of a small positive number is a large negative (bounded), while log of zero is `-inf` (unbounded).

**Acceptance:** `np.log()` calls never receive zero or negative inputs. A test with a DataFrame containing a zero price returns finite values (not `-inf`).

---

## Verification

- [ ] Run `pytest tests/ -k "feature or indicator or tail_risk"` — all pass
- [ ] Verify SemiRelMod_20 never exceeds 10.0 on extreme input
- [ ] Verify flat-bar input produces NaN (not inf) for all 30 guarded indicators
- [ ] Verify zero-price input produces finite values (not -inf) for log-based indicators
- [ ] Run the full pipeline on sample data and confirm no inf values in output
