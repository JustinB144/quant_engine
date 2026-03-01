# Audit Report: Subsystem 03 — Feature Engineering

> **Status:** Complete
> **Auditor:** Claude Opus 4.6
> **Date:** 2026-02-28
> **Spec:** `docs/audit/subsystem_specs/SPEC_AUDIT_03_FEATURE_ENGINEERING.md`

---

## Executive Summary

All **17 files** and **8,559 lines** in Subsystem 03 were reviewed line-by-line per spec. The overall architecture is sound: indicator computation pipelines are well-structured, structural analyzers have excellent numerical guards, and the expanding-window winsorization avoids look-ahead bias in feature clipping.

However, the audit found **1 P0 critical**, **2 P1 high**, **6 P2 medium**, and **6 P3 low** findings:

- **P0:** Two pivot indicators use `center=True` rolling windows, creating a true look-ahead leak in features classified as `CAUSAL`.
- **P1:** 16 emitted features are missing `FEATURE_METADATA` entries and silently default to `CAUSAL`; `TailRiskAnalyzer.semi_relative_modulus` can produce extreme values (~1e9) that survive winsorization.
- **P2:** 30+ indicator classes have unguarded division; `END_OF_DAY` features pass through the predictor causality gate without restriction; lazy-fallback failures are invisible in quiet mode; `compute_iv_shock_features` is orphaned; `rolling_vwap_*` metadata has no emitter; lookahead CI tests are too sparse.

Given the subsystem risk profile (`features/pipeline.py` hotspot score 16/21, `indicators/indicators.py` 14/21), P0 and P1 findings should be resolved before the next production train/predict cycle.

---

## Scope & Ledger (T1)

### File Coverage

| # | File | Lines | Risk Tier | Reviewed |
|---|---|---:|---|---|
| 1 | `features/pipeline.py` | 1,541 | HIGH | ✅ |
| 2 | `indicators/indicators.py` | 2,904 | HIGH | ✅ |
| 3 | `features/research_factors.py` | 985 | HIGH | ✅ |
| 4 | `indicators/eigenvalue.py` | 399 | MEDIUM | ✅ |
| 5 | `indicators/spectral.py` | 328 | MEDIUM | ✅ |
| 6 | `indicators/ssa.py` | 321 | MEDIUM | ✅ |
| 7 | `features/lob_features.py` | 311 | MEDIUM | ✅ |
| 8 | `indicators/ot_divergence.py` | 262 | MEDIUM | ✅ |
| 9 | `features/macro.py` | 244 | MEDIUM | ✅ |
| 10 | `features/intraday.py` | 243 | MEDIUM | ✅ |
| 11 | `features/harx_spillovers.py` | 242 | MEDIUM | ✅ |
| 12 | `indicators/tail_risk.py` | 240 | MEDIUM | ✅ |
| 13 | `features/version.py` | 168 | LOW | ✅ |
| 14 | `features/wave_flow.py` | 144 | LOW | ✅ |
| 15 | `features/options_factors.py` | 134 | LOW | ✅ |
| 16 | `indicators/__init__.py` | 89 | LOW | ✅ |
| 17 | `features/__init__.py` | 4 | LOW | ✅ |

**Total: 8,559/8,559 lines reviewed** (100% coverage).

### Spec vs Actual Discrepancies

| Item | Spec Value | Actual Value | Impact |
|---|---|---|---|
| `indicators/__init__.py` lines | 14 | 89 | None — file grew since spec was written |
| `features/__init__.py` lines | 34 | 4 | None — file was trimmed |
| Indicator classes | 92 | 91 | Off by 1 — audit used actual count |
| Pipeline indicator imports | 87 | 86 (top-level: 76 classes, plus 5 conditional analyzer imports, plus 5 from sub-feature modules) | Off by 1 |

Total line count of 8,559 is confirmed exact via `wc -l`.

### FEATURE_METADATA Inventory

`features/pipeline.py:87-411` — **262 entries** total:

| Causality Type | Count | Examples |
|---|---:|---|
| `CAUSAL` | 246 | ATR_14, RSI_14, TSMom_lag21, OFI_20, SpectralHFE_252, ... |
| `END_OF_DAY` | 13 | intraday_vol_ratio, vwap_deviation, trade_arrival_rate, ... |
| `RESEARCH_ONLY` | 3 | relative_mom_10, relative_mom_20, relative_mom_60 |

### Indicator Class Inventory

`indicators/indicators.py` — **91 concrete subclasses** of `Indicator(ABC)`:
- 76 imported at top level by `pipeline.py` (lines 21-64)
- 5 conditional analyzers imported lazily (Spectral, SSA, TailRisk, OptimalTransport, Eigenvalue)
- 5 exported by `indicators/__init__.py` but NOT consumed by pipeline (AnchoredVWAP, PriceVsAnchoredVWAP, MultiVWAPPosition, Beast666Proximity, Beast666Distance)
- 5 additional: `StochasticD`, `RVOL` are imported but `RVOL` is functionally identical to `VolumeRatio` (see F-11)

---

## Invariant Verification Summary

| Invariant | Status | Evidence |
|---|---|---|
| 100% line coverage across subsystem files | **PASS** | `wc -l` over Subsystem 3 file list = 8,559 lines |
| Every emitted feature has explicit metadata/type mapping | **FAIL** | 16 emitted feature names absent from `FEATURE_METADATA` (F-02) |
| Causality guard blocks forward-looking features | **FAIL** | `PivotHi_5_5` / `PivotLo_5_5` are forward-looking but typed `CAUSAL` (F-01) |
| Indicator outputs are numerically stable and schema-consistent | **PARTIAL FAIL** | 30+ indicators have unguarded division (F-04); TailRisk produces ~1e9 values (F-03); inf→NaN conversion at pipeline line 1132 prevents inf from reaching models |
| Structural analyzer interfaces are stable | **PASS** | `compute_all()` keys for spectral/SSA/tail/OT/eigen all match pipeline mapping |
| Lazy fallback behavior is graceful and observable | **PARTIAL FAIL** | Graceful fallback exists; observability is gated by `verbose` in key paths (F-06) |
| High-risk feature boundaries are explicitly validated | **PASS with findings** | `models_to_features_6` and `features_to_indicators_8` validated; both have critical/high caveats |

---

## Findings

### F-01 — Pivot indicators leak future data but are classified as `CAUSAL` [P0 CRITICAL]

**Invariant violated:** Causality contract (`models_to_features_6`).

**Proof:**
- `indicators/indicators.py:1031` (`PivotHigh`) and `indicators/indicators.py:1067` (`PivotLow`) use `rolling(window=pivot_window, center=True)`, which requires `right_bars=5` future bars to classify a pivot at the current timestamp.
- `features/pipeline.py:478` includes both indicators in the default feature set.
- `features/pipeline.py:149-150` classifies both as `{"type": "CAUSAL"}`.
- Predictor causality gate only blocks `RESEARCH_ONLY` (`models/predictor.py:216-223`), so these columns pass through to live prediction.

**Downstream impact:** Leaks future price structure into training and inference. Invalidates backtest realism and live causality guarantees.

**Recommended fix:** Replace `center=True` with a strictly causal formulation (e.g., only use left bars), or reclassify as `RESEARCH_ONLY`.

---

### F-02 — 16 emitted features are missing `FEATURE_METADATA` and default to `CAUSAL` [P1 HIGH]

**Invariant violated:** Feature metadata completeness and explicit causality typing.

**Proof:**
- `get_feature_type()` defaults unknown names to `"CAUSAL"` (`features/pipeline.py:414-427`).
- `models/predictor.py:22,216-223` uses this function as the runtime causality gate.
- `compute_universe()` emits these features via lazy joins but none are registered:

| Feature | Source Module | Actual Causality |
|---|---|---|
| `avg_pairwise_corr` | `regime/correlation.py:208` | CAUSAL (correct) |
| `corr_regime` | `regime/correlation.py:209` | CAUSAL (correct) |
| `corr_z_score` | `regime/correlation.py:210` | CAUSAL (correct) |
| `harx_spillover_from` | `features/harx_spillovers.py:227` | CAUSAL (correct) |
| `harx_spillover_to` | `features/harx_spillovers.py:228` | CAUSAL (correct) |
| `harx_net_spillover` | `features/harx_spillovers.py:229` | CAUSAL (correct) |
| `macro_vix` | `features/macro.py:33` | CAUSAL (correct) |
| `macro_vix_mom` | `features/macro.py:242` | CAUSAL (correct) |
| `macro_term_spread` | `features/macro.py:34` | CAUSAL (correct) |
| `macro_term_spread_mom` | `features/macro.py:242` | CAUSAL (correct) |
| `macro_credit_spread` | `features/macro.py:35` | CAUSAL (correct) |
| `macro_credit_spread_mom` | `features/macro.py:242` | CAUSAL (correct) |
| `macro_initial_claims` | `features/macro.py:36` | CAUSAL (correct) |
| `macro_initial_claims_mom` | `features/macro.py:242` | CAUSAL (correct) |
| `macro_consumer_sentiment` | `features/macro.py:37` | CAUSAL (correct) |
| `macro_consumer_sentiment_mom` | `features/macro.py:242` | CAUSAL (correct) |

**Current risk:** All 16 happen to be genuinely CAUSAL, so no active data leakage exists today. However, the default-to-CAUSAL policy means any future non-causal feature added to these modules would silently bypass the truth layer.

**Recommended fix:** Register all 16 features in `FEATURE_METADATA`. Consider changing the default in `get_feature_type()` to raise an error or return a safe default like `"RESEARCH_ONLY"` for unknown features.

---

### F-03 — `TailRiskAnalyzer.semi_relative_modulus` can produce extreme values (~1e9) [P1 HIGH]

**Invariant violated:** Indicator numerical stability.

**Proof:**
- `indicators/tail_risk.py:180-184`:
  ```python
  if up_var > 1e-15:
      srm[i] = np.sqrt(down_var / up_var)
  else:
      # No upside variance — extreme downside dominance
      srm[i] = np.sqrt(down_var) / 1e-10 if down_var > 0 else 0.0
  ```
- When `up_var <= 1e-15` but `down_var > 0` (which occurs during sustained drawdown periods):
  - `down_var = 0.0001` → `sqrt(0.0001) / 1e-10 = 1e6`
  - `down_var = 0.01` → `sqrt(0.01) / 1e-10 = 1e9`
- Pipeline converts `inf → NaN` at line 1132, but these are finite values — they survive.
- Expanding-window winsorization at line 1152 clips to [1st, 99th] percentile, but if multiple extreme values exist, the 99th percentile itself is inflated.
- `SemiRelMod_20` maps to this output (`features/pipeline.py:204`) and is classified `CAUSAL`, so it reaches the model.

**Downstream impact:** Extreme feature values (~1e6 to ~1e9) can dominate tree splits or gradient updates. The feature's magnitude is 6-9 orders of magnitude larger than other features.

**Recommended fix:** Cap the fallback branch (e.g., `min(np.sqrt(down_var) / 1e-10, 100.0)`) or use a more stable asymmetry measure when upside variance is negligible.

---

### F-04 — 30+ indicator classes have unguarded division [P2 MEDIUM]

**Invariant violated:** Indicator numerical stability (defensive coding).

**Proof:** 30 unique indicator classes in `indicators/indicators.py` contain divisions where the denominator can be zero with no explicit guard. Key examples:

| Class | Line | Denominator | Trigger Condition |
|---|---|---|---|
| `VolumeRatio` | 490 | `avg_vol` | Volume = 0 for entire period |
| `RVOL` | 826 | `avg_vol` | Same as above |
| `Stochastic` | 258 | `high_max - low_min` | Flat bar (High == Low) |
| `WilliamsR` | 300 | `high_max - low_min` | Flat bar |
| `CandleBody` | 615 | `range_` | Flat bar |
| `AccumulationDistribution` | 900 | `High - Low` | Flat bar |
| `ADX` | 435-438 | `atr`, `plus_di + minus_di` | Zero volatility |
| `VolatilityRegime` | 1302 | `vol_std` | Constant volatility |
| `VWAP` | 1332 | `Volume.rolling.sum()` | Zero volume |
| `MFI` | 554 | `negative_mf.sum()` | All bars positive |

Full list: NATR, BollingerBandWidth, RSI, ROC, Stochastic, WilliamsR, CCI, PriceVsSMA, SMASlope, ADX (3×), VolumeRatio, RVOL, MFI, CandleBody, GapPercent, DistanceFromHigh, DistanceFromLow, NetVolumeTrend, AccumulationDistribution, TrendStrength, ATRTrailingStop, ATRChannel, RiskPerATR, VolatilityRegime, VWAP, PriceVsVWAP, VWAPBands, PriceVsPOC, ValueAreaPosition, ShannonEntropy, FractalDimension, DominantCycle.

**Mitigating factor:** Pipeline converts `inf → NaN` at line 1132-1133, so inf values don't reach models. However, the conversion is silent — information is lost and replaced by median fill in the predictor.

**Note — well-guarded indicators (positive examples):** ZScore (`.replace(0, np.nan)`), VarianceRatio (`.replace(0, np.nan)`), AmihudIlliquidity (`.replace(0, np.nan)`), RollSpread (`.replace(0, np.nan)`), HurstExponent (`if S > 1e-12`), Autocorrelation (`if den > 1e-12`), Beast666Proximity (`+ 1e-12` epsilon), VolTermStructure (`.replace(0, np.nan)`), ValueAreaHigh/Low/POC (`if total_vol == 0: continue`).

**Recommended fix:** Add `.replace(0, np.nan)` or epsilon guards to the 30 unguarded indicators, following the patterns already used by the guarded ones.

---

### F-05 — `END_OF_DAY` features are not blocked by predictor causality gate [P2 MEDIUM]

**Invariant violated:** Causality enforcement completeness (`models_to_features_6`).

**Proof:**
- `FEATURE_METADATA` defines 13 features as `END_OF_DAY` (6 intraday + 7 LOB).
- `models/predictor.py:213-225` only blocks `RESEARCH_ONLY` features:
  ```python
  if TRUTH_LAYER_ENFORCE_CAUSALITY:
      research_only = {
          col for col in features.columns
          if get_feature_type(col) == "RESEARCH_ONLY"
      }
  ```
- `END_OF_DAY` features require same-day close data (unavailable during intraday prediction).
- No check exists for whether the system is running in daily vs intraday mode.

**Current risk:** Low if the system only makes end-of-day predictions. Would become a live data leak if the system were extended to intraday prediction.

**Recommended fix:** Add `END_OF_DAY` to the predictor causality gate, or document the daily-only constraint as an explicit invariant.

---

### F-06 — Lazy-fallback failures are non-observable in quiet mode [P2 MEDIUM]

**Invariant violated:** Explicit observability for degraded feature blocks.

**Proof:**
- Multiple optional blocks catch exceptions and only report via `if verbose: print(...)`:
  - HARX (`features/pipeline.py:1294-1297`)
  - Correlation regime (`features/pipeline.py:1320-1323`)
  - Eigenvalue (`features/pipeline.py:1380-1382`)
  - Macro (`features/pipeline.py:1402-1404`)
  - Intraday/LOB (`features/pipeline.py:1451-1453`, `1487-1489`, `1499-1501`)
- Additionally, structural analyzers 1-4 (`compute_structural_features()` lines 769-848) catch bare `Exception`, which swallows bugs like `TypeError`, `KeyError`, `AttributeError` alongside expected `ImportError`.
- Analyzer 5 (`EigenvalueAnalyzer`, line 1337) catches the more specific `(ImportError, ValueError, RuntimeError)` — inconsistent with the other four.
- When features silently disappear, predictor's median-fill logic (`models/predictor.py:40-47`) substitutes imputed values without warning.

**Downstream impact:** Model can silently degrade from real signals to imputed medians without any operator visibility.

**Recommended fix:** Use `logging.warning()` (not gated by `verbose`) for all fallback paths. Narrow structural analyzer exception catches to match analyzer 5's pattern.

---

### F-07 — `rolling_vwap_*` metadata entries have no pipeline emitter [P2 MEDIUM]

**Invariant violated:** Feature contract consistency (metadata ↔ runtime schema).

**Proof:**
- `FEATURE_METADATA` includes `rolling_vwap_20` (`features/pipeline.py:400`) and `rolling_vwap_deviation_20` (`features/pipeline.py:401`).
- `compute_rolling_vwap()` exists in `features/intraday.py:203` but is never called from `FeaturePipeline.compute()` or `FeaturePipeline.compute_universe()`.
- No call site references `compute_rolling_vwap` anywhere in the pipeline.

**Downstream impact:** Metadata implies availability of 2 columns that never exist in produced feature frames. Any downstream code relying on these entries will find no matching data.

**Recommended fix:** Either wire `compute_rolling_vwap` into the pipeline, or remove the 2 metadata entries.

---

### F-08 — Lookahead CI tests are too sparse to catch pivot-class leaks [P2 MEDIUM]

**Invariant violated:** Test adequacy for causality regressions.

**Proof:**
- `tests/test_lookahead_detection.py` checks limited timestamps per truncation (single boundary row or interior checkpoint).
- Full prefix-replay across all timestamps would detect the `PivotHigh`/`PivotLow` `center=True` mismatches that current tests miss (see F-01).

**Recommended fix:** Expand causality tests to prefix-replay across multiple bars, not single checkpoints. Add explicit test for `center=True` rolling window usage.

---

### F-09 — `compute_iv_shock_features` is orphaned from pipeline [P2 MEDIUM]

**Invariant violated:** Feature surface completeness.

**Proof:**
- `features/options_factors.py:92` defines `compute_iv_shock_features()` producing 4 features: `iv_shock_{window}`, `iv_shock_magnitude_{window}`, `vol_regime_shift_{window}`, `iv_compression_{window}`.
- `features/pipeline.py:70` imports only `compute_option_surface_factors` — not `compute_iv_shock_features`.
- No call site in `pipeline.py` invokes this function.
- Tests exist (`tests/test_feature_fixes.py:117-147`) but only test the function in isolation.

**Downstream impact:** 4 features that were presumably designed for production are never computed. Dead production code indicates incomplete integration or abandoned work.

**Recommended fix:** Either integrate into pipeline (with FEATURE_METADATA entries), or move to a research/experimental module.

---

### F-10 — `features/version.py` is dead code [P3 LOW]

**Proof:**
- `features/version.py` (168 lines) defines `FeatureVersion` and `FeatureRegistry` classes.
- No file in the codebase imports from `features.version`.
- Grep for `from.*version import|from.*features\.version|import.*version` across all `*.py` files returns zero matches.

**Recommended fix:** Remove or mark as experimental.

---

### F-11 — `VolumeRatio` and `RVOL` are functionally identical [P3 LOW]

**Proof:**
- `VolumeRatio` (`indicators/indicators.py:475-490`):
  ```python
  avg_vol = df['Volume'].rolling(window=self.period).mean()
  return df['Volume'] / avg_vol
  ```
- `RVOL` (`indicators/indicators.py:808-827`):
  ```python
  avg_vol = df['Volume'].rolling(window=self.period).mean()
  rvol = df['Volume'] / avg_vol
  return rvol
  ```
- Both use `period=20` by default, producing `VolRatio_20` and `RVOL_20` — identical values under different names.
- Both are imported and instantiated by the pipeline, creating redundant feature columns.

**Recommended fix:** Remove one and alias the other, or differentiate the implementation (e.g., RVOL could normalize by time-of-day volume profile).

---

### F-12 — 5 analyzer files have dead `import pandas as pd` [P3 LOW]

**Proof:** All 5 structural analyzers accept `np.ndarray` inputs and return `Dict[str, np.ndarray]`. None use pandas:
- `indicators/spectral.py:12`
- `indicators/ssa.py:12`
- `indicators/tail_risk.py:12`
- `indicators/ot_divergence.py:13`
- `indicators/eigenvalue.py:16`

---

### F-13 — 5 indicator classes exported but not consumed by pipeline [P3 LOW]

**Proof:** `indicators/__init__.py` exports these classes, but `pipeline.py` never imports or instantiates them:
- `AnchoredVWAP`
- `PriceVsAnchoredVWAP`
- `MultiVWAPPosition`
- `Beast666Proximity`
- `Beast666Distance`

**Note:** These may be intended for external consumers or future integration.

---

### F-14 — `INDICATOR_ALIASES` is redundant [P3 LOW]

**Proof:**
- `indicators/indicators.py:2885-2896` defines `INDICATOR_ALIASES` mapping 10 common names to classes.
- All 10 aliases (e.g., `"rsi"` → `RSI`, `"macd"` → `MACD`) are already returned by `get_all_indicators()`.
- `create_indicator()` does NOT consult `INDICATOR_ALIASES` — it only searches the `get_all_indicators()` registry.

---

### F-15 — Zero dedicated indicator tests [P3 LOW]

**Proof:**
- Grep for test files targeting indicators: `tests/test_indicator*`, `tests/test_indicators*` — no matches.
- 91 concrete indicator classes have no unit tests verifying their numerical output.
- Some indicators are tested indirectly through feature pipeline integration tests, but no class-level coverage exists.

---

## Interface Contract Verification (T5)

### Boundary: `models_to_features_6` — **FAIL**

| Check | Result |
|---|---|
| `get_feature_type` import/use in predictor (`models/predictor.py:22`) | ✅ |
| Runtime blocking of `RESEARCH_ONLY` features (`models/predictor.py:216-223`) | ✅ |
| Unknown feature typing behavior (`features/pipeline.py:414-427`) | ⚠️ Defaults to `CAUSAL` (F-02) |
| Metadata completeness against emitted universe columns | ❌ 16 missing entries (F-02) |
| Causality correctness for `CAUSAL` features | ❌ Pivot leak (F-01) |
| `END_OF_DAY` features blocked in non-daily modes | ❌ Not checked (F-05) |

### Boundary: `features_to_indicators_8` — **PARTIAL FAIL**

| Check | Result |
|---|---|
| Top-level indicator import set in pipeline (76 classes) | ✅ |
| 5 conditional analyzer imports with fallback | ✅ |
| 91 indicator subclasses instantiate correctly | ✅ |
| Structural analyzer `compute_all` key contracts match pipeline mapping | ✅ |
| Indicator numerical stability | ⚠️ 30+ unguarded divisions (F-04), extreme values in TailRisk (F-03) |
| Indicator causality assumptions | ❌ Broken for `PivotHigh`/`PivotLow` (F-01) |

### Boundary: `features_to_config_13` — **PASS**

| Check | Result |
|---|---|
| Pipeline imports 19 unique config constants | ✅ |
| `STRUCTURAL_FEATURES_ENABLED` gating present for structural blocks | ✅ |
| Config dependency edges consistent with contract matrix | ✅ |

Config constants consumed: `SPECTRAL_FFT_WINDOW`, `SPECTRAL_CUTOFF_PERIOD`, `SSA_WINDOW`, `SSA_EMBED_DIM`, `SSA_N_SINGULAR`, `JUMP_INTENSITY_WINDOW`, `JUMP_INTENSITY_THRESHOLD`, `WASSERSTEIN_WINDOW`, `WASSERSTEIN_REF_WINDOW`, `SINKHORN_EPSILON`, `SINKHORN_MAX_ITER`, `INTERACTION_PAIRS`, `FORWARD_HORIZONS`, `STRUCTURAL_FEATURES_ENABLED`, `EIGEN_CONCENTRATION_WINDOW`, `EIGEN_MIN_ASSETS`, `EIGEN_REGULARIZATION`, `BENCHMARK`, `LOOKBACK_YEARS`, `INTRADAY_MIN_BARS`.

### Boundary: `features_to_data_14` — **PASS with caveat**

| Check | Result |
|---|---|
| `load_ohlcv` lazy import (`features/pipeline.py:1528`) | ✅ |
| `WRDSProvider` conditional import (`features/pipeline.py:1413`) | ✅ |
| `load_intraday_ohlcv` conditional import (`features/pipeline.py:1458`) | ✅ |
| Failure observability in quiet mode | ⚠️ Caveat (F-06) |

### Boundary: `features_to_regime_15` — **PASS with caveat**

| Check | Result |
|---|---|
| Optional import of `CorrelationRegimeDetector` (`features/pipeline.py:1303`) | ✅ |
| Output columns consumed through date-level join | ✅ |
| Correlation feature metadata registration | ⚠️ Missing (F-02) |

---

## Numerical Integrity Pass (T3)

### Indicator Surface

- `Indicator` subclasses found: **91**
- Calculation failures: **0**
- Non-`Series` returns: **0**
- Length mismatches: **0**
- Unguarded divisions: **30 classes** (see F-04)
- Unguarded `np.log()` calls: **5 classes** (ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility, HurstExponent, DFA) — log of price ratios which are positive in practice but lack explicit positivity guards

### Structural Analyzer Surface

`compute_all()` key contracts verified stable:

| Analyzer | Keys |
|---|---|
| SpectralAnalyzer | `hf_energy`, `lf_energy`, `spectral_entropy`, `dominant_period`, `spectral_bandwidth` |
| SSADecomposer | `trend_strength`, `oscillatory_strength`, `singular_entropy`, `noise_ratio` |
| TailRiskAnalyzer | `jump_intensity`, `expected_shortfall`, `vol_of_vol`, `semi_relative_modulus`, `extreme_return_pct` |
| OptimalTransportAnalyzer | `wasserstein_distance`, `sinkhorn_divergence` |
| EigenvalueAnalyzer | `eigenvalue_concentration`, `effective_rank`, `avg_correlation_stress`, `condition_number` |

All 20 keys are hardcoded strings matching `FEATURE_METADATA` mappings in `pipeline.py`. Analyzers use excellent internal guards:
- SpectralAnalyzer: `np.clip`, `max(..., 1e-10)`, `if len(returns) < window: return {}`
- SSADecomposer: `if n < window: return {}`, `total / max(total, 1e-10)`
- OptimalTransportAnalyzer: log-domain Sinkhorn for numerical stability
- EigenvalueAnalyzer: `eigenvalues = np.maximum(eigenvalues, 0)`, `max(n, 1)` guards

### Pipeline Inf/NaN Handling

- `features/pipeline.py:1132-1133`: `features.replace([np.inf, -np.inf], np.nan)` — converts all infinities to NaN after indicator computation
- `features/pipeline.py:1142-1143`: Same replacement after interaction features
- `features/pipeline.py:961-976`: `_winsorize_expanding()` clips to [1st, 99th] percentile via expanding window (no look-ahead)
- `models/predictor.py:40-47`: Median fill for remaining NaN values

---

## Data/Regime Lazy Import Pass (T4)

### Lazy Import Inventory

| Import | Location | Fallback Behavior | Exception Scope |
|---|---|---|---|
| `SpectralAnalyzer` | pipeline.py:769 | Silent skip (verbose print) | `Exception` (too broad) |
| `SSADecomposer` | pipeline.py:789 | Silent skip (verbose print) | `Exception` (too broad) |
| `TailRiskAnalyzer` | pipeline.py:809 | Silent skip (verbose print) | `Exception` (too broad) |
| `OptimalTransportAnalyzer` | pipeline.py:836 | Silent skip (verbose print) | `Exception` (too broad) |
| `EigenvalueAnalyzer` | pipeline.py:1337 | Silent skip (verbose print) | `(ImportError, ValueError, RuntimeError)` ✅ |
| `compute_harx_spillovers` | pipeline.py:1284 | Silent skip (verbose print) | `Exception` (too broad) |
| `CorrelationRegimeDetector` | pipeline.py:1303 | Silent skip (verbose print) | `(ImportError, ValueError, RuntimeError)` ✅ |
| `MacroFeatureProvider` | pipeline.py:1389 | Silent skip (verbose print) | `Exception` (too broad) |
| `WRDSProvider` | pipeline.py:1413 | Fall through to local cache | `ImportError` ✅ |
| `compute_intraday_features` | pipeline.py:1424 | Silent skip (verbose print) | `Exception` (too broad) |
| `compute_lob_features` | pipeline.py:1425/1459 | Silent skip (verbose print) | `Exception` (too broad) |
| `load_intraday_ohlcv` | pipeline.py:1458 | Silent skip (verbose print) | `Exception` (too broad) |
| `load_ohlcv` | pipeline.py:1528 | Hard fail (no try/except) | N/A |
| `STRUCTURAL_FEATURES_ENABLED` | pipeline.py:1120/1326 | Default `False` | `ImportError` ✅ |

**Pattern:** 9 of 14 lazy import paths catch bare `Exception`, which can silently swallow genuine bugs (TypeError, KeyError, AttributeError). Only 3 use appropriately scoped exception tuples.

### Research Factors Lazy Imports

`features/research_factors.py` has 3 lazy external library imports:
- `dtaidistance` (DTW features) — try/except with callable disabled
- `tslearn` (DTW normalization) — try/except with callable disabled
- `iisignature` (path signatures) — try/except with callable disabled

All three follow the correct pattern: import at module level, set availability flag, skip computation if unavailable.

---

## Findings Summary & Risk Disposition (T6)

### By Severity

| Severity | Count | IDs |
|---|---:|---|
| P0 Critical | 1 | F-01 |
| P1 High | 2 | F-02, F-03 |
| P2 Medium | 6 | F-04, F-05, F-06, F-07, F-08, F-09 |
| P3 Low | 6 | F-10, F-11, F-12, F-13, F-14, F-15 |
| **Total** | **15** | |

### Defect Matrix

| ID | Severity | Category | Files Affected | Feature/Contract Impact |
|---|---|---|---|---|
| F-01 | P0 | Causality leak | indicators.py:1031,1067; pipeline.py:149-150,478 | `PivotHi_5_5`, `PivotLo_5_5` leak 5 future bars |
| F-02 | P1 | Metadata gap | pipeline.py:414-427; macro.py; harx_spillovers.py; regime/correlation.py | 16 features bypass explicit typing |
| F-03 | P1 | Numerical instability | tail_risk.py:180-184 | `SemiRelMod_20` can be ~1e9 |
| F-04 | P2 | Numerical instability | indicators.py (30 classes) | 30+ features produce inf→NaN silently |
| F-05 | P2 | Causality gate gap | predictor.py:213-225 | 13 END_OF_DAY features not gated |
| F-06 | P2 | Observability | pipeline.py (7 try/except blocks) | Silent degradation in quiet mode |
| F-07 | P2 | Contract drift | pipeline.py:400-401; intraday.py:203 | 2 phantom metadata entries |
| F-08 | P2 | Test adequacy | test_lookahead_detection.py | Pivot leaks pass CI |
| F-09 | P2 | Dead integration | options_factors.py:92 | 4 features never computed |
| F-10 | P3 | Dead code | version.py (168 lines) | Entire file unused |
| F-11 | P3 | Redundancy | indicators.py:475,808 | VolRatio_20 ≡ RVOL_20 |
| F-12 | P3 | Dead imports | 5 analyzer files | `import pandas as pd` unused |
| F-13 | P3 | Unused exports | indicators/__init__.py | 5 classes exported, not consumed |
| F-14 | P3 | Redundancy | indicators.py:2885-2896 | INDICATOR_ALIASES unused |
| F-15 | P3 | Test gap | — | 0 dedicated indicator unit tests |

### Recommended Remediation Order

1. **F-01 (P0):** Replace `center=True` with strictly causal pivot formulation. Reclassify if needed. Add regression test.
2. **F-02 (P1):** Register all 16 missing features in `FEATURE_METADATA`. Consider changing unknown-feature default to raise or return `RESEARCH_ONLY`.
3. **F-03 (P1):** Cap `semi_relative_modulus` fallback branch. Add unit test with extreme input.
4. **F-06 (P2):** Promote all fallback failures to `logging.warning()`. Narrow bare `Exception` catches to specific exception tuples.
5. **F-04 (P2):** Add division guards to 30 indicator classes following existing patterns (`.replace(0, np.nan)` or epsilon).
6. **F-05 (P2):** Add `END_OF_DAY` to predictor causality gate, or document daily-only constraint.
7. **F-08 (P2):** Expand causality tests to prefix-replay across all bars.
8. **F-07 (P2):** Wire `compute_rolling_vwap` into pipeline, or remove phantom metadata.
9. **F-09 (P2):** Integrate `compute_iv_shock_features` or move to experimental.
10. **F-10–F-15 (P3):** Address as part of regular cleanup.

---

## Acceptance Criteria Checklist

| Criterion | Status |
|---|---|
| 100% of lines in all 17 files reviewed | ✅ 8,559/8,559 |
| Every emitted feature has explicit metadata/type mapping | ❌ 16 missing (F-02) |
| Indicator outputs and names stable for downstream consumers | ✅ (all 91 classes produce consistent output names) |
| High-risk boundaries (`models_to_features_6`, `features_to_indicators_8`) explicitly validated | ✅ (both validated; both with critical/high findings documented) |

---

## Audit Notes

- No code changes were made as part of this audit.
- Previous audit by Codex (GPT-5) found 5 findings (F-01 through F-05 in their numbering). This audit confirms all 5 and adds 10 additional findings through deeper line-by-line review.
- Spec verification command in T5 for `DEPENDENCY_EDGES.json` required adaptation because `.symbols_imported` is an array in the current dataset.
- 5 analyzer files (`spectral.py`, `ssa.py`, `tail_risk.py`, `ot_divergence.py`, `eigenvalue.py`) have uniformly excellent internal numerical guards — they represent best practices that should be applied to the main `indicators.py` classes.
