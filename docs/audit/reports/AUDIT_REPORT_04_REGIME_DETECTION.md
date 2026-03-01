# Audit Report: Subsystem 04 — Regime Detection

> **Status:** Complete
> **Auditor:** Claude (Opus 4.6)
> **Date:** 2026-02-28
> **Spec:** `docs/audit/subsystem_specs/SPEC_AUDIT_04_REGIME_DETECTION.md`

---

## Executive Summary

All **13 files** and **4,420 lines** in the `regime_detection` subsystem have been reviewed line-by-line. The subsystem is well-architected — the multi-engine detector with fallbacks, version-locked ShockVector schema, and entropy-based uncertainty gating are carefully implemented with defensive programming throughout. However, **0 P0 critical findings**, **3 P1 high findings**, **5 P2 medium findings**, and **5 P3 low findings** were identified. The P1 findings relate to a forward-compatibility gap in `ShockVector.from_dict`, a discrepancy between `compute_shock_vectors` and `detect_with_shock_context` in structural feature production, and an unused config import that indicates dead integration code.

---

## Scope & Ledger (T1)

### File Coverage

| # | File | Lines | Risk Tier | Reviewed |
|---|---|---:|---|---|
| 1 | `regime/detector.py` | 940 | HIGH (15/21) | Yes |
| 2 | `regime/hmm.py` | 661 | MEDIUM | Yes |
| 3 | `regime/shock_vector.py` | 494 | HIGH (12/21) | Yes |
| 4 | `regime/bocpd.py` | 451 | MEDIUM | Yes |
| 5 | `regime/jump_model_pypi.py` | 420 | MEDIUM | Yes |
| 6 | `regime/consensus.py` | 273 | LOW | Yes |
| 7 | `regime/confidence_calibrator.py` | 251 | LOW | Yes |
| 8 | `regime/online_update.py` | 245 | LOW | Yes |
| 9 | `regime/jump_model_legacy.py` | 242 | LOW | Yes |
| 10 | `regime/correlation.py` | 213 | LOW | Yes |
| 11 | `regime/uncertainty_gate.py` | 180 | HIGH (12/21) | Yes |
| 12 | `regime/__init__.py` | 41 | LOW | Yes |
| 13 | `regime/jump_model.py` | 9 | LOW | Yes |

**Total: 4,420 lines reviewed.** 13/13 files — 100% coverage. Matches spec expectation exactly.

### Downstream Consumer Files Reviewed (Read-Only)

| File | Why |
|---|---|
| `backtest/engine.py` | ShockVector + UncertaintyGate primary consumer |
| `backtest/execution.py` | ShockVector type-checking import, ShockModePolicy |
| `risk/position_sizer.py` | UncertaintyGate consumer for Kelly sizing |
| `autopilot/engine.py` | RegimeDetector + UncertaintyGate consumer |

---

## Invariant Verification Summary

| Invariant | Status | Evidence |
|---|---|---|
| Canonical regime IDs (0-3) | **PASS** | config.py:217-222 defines `REGIME_NAMES {0: trending_bull, 1: trending_bear, 2: mean_reverting, 3: high_volatility}`. detector.py, hmm.py, shock_vector.py all use consistent 0-3 mapping. Default/fallback is always 2 (mean_reverting). |
| Regime label consistency across engines | **PASS** | Rule-based (detector.py:180-198), HMM (hmm.py:612-660 via `map_raw_states_to_regimes`), Jump (detector.py:336 reuses `map_raw_states_to_regimes`), and Ensemble (detector.py:390-509) all produce regime IDs in [0, 3]. All use the same state-mapping infrastructure. |
| ShockVector schema version lock | **PASS** | shock_vector.py:26 defines `_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})`. `__post_init__` (line 88) raises `ValueError` for unsupported versions. `from_dict` (line 133) passes through `cls(**d)` which triggers `__post_init__`. Version is enforced at both construction and deserialization. |
| ShockVector 13-field schema | **PASS** | All 13 fields verified at shock_vector.py:72-84: `schema_version`, `timestamp`, `ticker`, `hmm_regime`, `hmm_confidence`, `hmm_uncertainty`, `bocpd_changepoint_prob`, `bocpd_runlength`, `jump_detected`, `jump_magnitude`, `structural_features`, `transition_matrix`, `ensemble_model_type`. |
| ShockVector to_dict/from_dict round-trip | **PARTIAL FAIL** | `to_dict` correctly excludes `transition_matrix` and converts timestamp to ISO string. `from_dict` correctly parses ISO timestamp and drops `transition_matrix`. However, `from_dict` does NOT filter unknown keys — see F-01. |
| UncertaintyGate config consistency | **PASS** | All 3 consumers (backtest/engine.py:278, risk/position_sizer.py:105, autopilot/engine.py:1672) instantiate `UncertaintyGate()` with no arguments, meaning all use the same config defaults: `REGIME_UNCERTAINTY_ENTROPY_THRESHOLD=0.50`, `REGIME_UNCERTAINTY_STRESS_THRESHOLD=0.80`, `REGIME_UNCERTAINTY_SIZING_MAP={0.0:1.0, 0.5:0.95, 1.0:0.85}`, `REGIME_UNCERTAINTY_MIN_MULTIPLIER=0.80`. |
| Confidence values in [0, 1] | **PASS** | Rule-based confidence clipped to [0.3, 1.0] (detector.py:203-210). HMM confidence = max posterior probability, inherently in [0, 1] (detector.py:297). Jump model same pattern (detector.py:357). Ensemble vote fraction inherently in [0, 1]. ShockVector `__post_init__` clips both confidence and uncertainty to [0, 1]. |
| BOCPD graceful degradation | **PASS** | detector.py:116-125: BOCPD init failure logs warning and sets `enable_bocpd = False`. shock_vector.py:378-393: BOCPD batch failure caught, defaults to zeros. detector.py:752-757: per-ticker BOCPD failure caught. |

---

## Contract Verification (T3, T4)

### Boundary: `backtest_to_regime_shock_1` — **CONFIRMED COMPATIBLE**

| Check | Result |
|---|---|
| `compute_shock_vectors` signature matches INTERFACE_CONTRACTS.yaml | **PASS** — 9 parameters, all defaults match: `bocpd_hazard_lambda=1/60`, `bocpd_hazard_func="constant"`, `bocpd_max_runlength=200`, `jump_sigma_threshold=2.5`, `vol_lookback=20`. Return type `Dict[timestamp, ShockVector]` confirmed. |
| `ShockVector` fields accessed by backtest/engine.py are present | **PASS** — Consumer accesses: `hmm_uncertainty`, `bocpd_changepoint_prob`, `structural_features.get("drift_score")`, `structural_features.get("systemic_stress")`, `is_shock_event()`. All present in schema. |
| `ShockVector` fields accessed by backtest/execution.py are present | **PASS** — Consumer accesses: `is_shock_event()`, `hmm_uncertainty`. Both present. |
| `ShockVectorValidator` validation rules match schema | **PASS** — 10 validation checks cover all 13 fields with appropriate type and range constraints. |

### Boundary: `backtest_to_regime_uncertainty_2` — **CONFIRMED COMPATIBLE**

| Check | Result |
|---|---|
| `UncertaintyGate.compute_size_multiplier` contract | **PASS** — Returns float in [`min_multiplier`, 1.0]. backtest/engine.py (line ~1107) and risk/position_sizer.py (line ~263) both call this correctly with a float uncertainty input. |
| `UncertaintyGate.apply_uncertainty_gate` contract | **PASS** — Returns scaled ndarray. autopilot/engine.py (line ~1672) calls this with weights array and scalar uncertainty. |
| `UncertaintyGate.should_assume_stress` contract | **PASS** — Returns bool. Not directly called by consumers in the current codebase (downstream wiring ready but not yet active). |
| Threshold consistency across consumers | **PASS** — All 3 consumers use default config values (no overrides). |

### Boundary: `features_to_regime_15` — **CONFIRMED COMPATIBLE**

| Check | Result |
|---|---|
| `CorrelationRegimeDetector` conditional import | **PASS** — features/pipeline.py:1303 imports inside try/except. Failure is graceful. |
| `get_correlation_features` return schema | **PASS** — Returns DataFrame with `avg_pairwise_corr`, `corr_regime`, `corr_z_score`. These become feature columns. |

### Boundary: `risk_to_regime_22` — **CONFIRMED COMPATIBLE**

| Check | Result |
|---|---|
| `UncertaintyGate` import at risk/position_sizer.py:27 | **PASS** — Top-level import, used at lines 105, 263-270, 974-1019. |
| Sizing multiplier semantics | **PASS** — Multiplier in [0.80, 1.0] applied to both raw and half-Kelly fractions. |

### Boundary: `autopilot_to_regime_29` — **CONFIRMED COMPATIBLE**

| Check | Result |
|---|---|
| `RegimeDetector` import at autopilot/engine.py:60 | **PASS** — Top-level import. Used in `_build_regimes()` (line ~242) via `detector.regime_features()`. |
| `UncertaintyGate` import at autopilot/engine.py:61 | **PASS** — Top-level import. Used in portfolio weight adjustment (line ~1672). |
| Regime output columns consumed | **PASS** — autopilot expects `regime` (int), `regime_confidence` (float), `regime_prob_*` columns. All produced by `regime_features()`. |

---

## Findings

### F-01 — `ShockVector.from_dict` rejects unknown keys, breaking forward compatibility [P1 HIGH]

**Invariant at risk:** Schema versioning promise — "V1 consumers can safely ignore fields added in V2+."

**Proof:**
- shock_vector.py:7-8 docstring states: "Schema versioning ensures backward compatibility: V1 consumers can safely ignore fields added in V2+."
- shock_vector.py:128-133: `from_dict` does `d = dict(data)`, pops only `transition_matrix`, then calls `cls(**d)`. If the dict contains any key not in the dataclass definition (e.g., a field added in a hypothetical V2 schema), `cls(**d)` will raise `TypeError: __init__() got an unexpected keyword argument`.
- This means a V1 `from_dict` **cannot** deserialize a V2-serialized dict that contains new fields.

**Downstream impact:** Currently no impact because only schema version "1.0" exists. However, the first schema version bump will break deserialization for any code that hasn't been updated, contradicting the documented compatibility promise.

**Recommended fix:**
```python
@classmethod
def from_dict(cls, data: Dict) -> "ShockVector":
    d = dict(data)
    ts = d.get("timestamp")
    if isinstance(ts, str):
        d["timestamp"] = datetime.fromisoformat(ts)
    d.pop("transition_matrix", None)
    # Filter to known fields for forward compatibility
    known_fields = {f.name for f in fields(cls)}
    d = {k: v for k, v in d.items() if k in known_fields}
    return cls(**d)
```

---

### F-02 — `compute_shock_vectors` and `detect_with_shock_context` produce different structural features [P1 HIGH]

**Invariant at risk:** Structural feature consistency between batch and single-bar ShockVector construction paths.

**Proof:**
- **`compute_shock_vectors`** (shock_vector.py:469-472) always produces exactly 2 structural features:
  - `drift_score` — computed from `|price - SMA| / (ATR * sqrt(lookback))`
  - `systemic_stress` — computed from expanding vol percentile
- **`detect_with_shock_context`** (detector.py:778-790) produces up to 4 different structural features:
  - `spectral_entropy` — from `SpectralEntropy_252` column
  - `ssa_trend_strength` — from `SSATrendStr_60` column
  - `jump_intensity` — from `JumpIntensity_20` column
  - `eigenvalue_concentration` — from `EigenConcentration_60` column
  - Notably, it does **NOT** produce `drift_score` or `systemic_stress`.

- **backtest/engine.py** calls `compute_shock_vectors` (line 848-886 in the `run()` method) to pre-compute ShockVectors. It then accesses `shock.structural_features.get("drift_score")` and `shock.structural_features.get("systemic_stress")` (lines 446/579 and 447/580).
- If any consumer were to call `detect_with_shock_context` instead, those `.get("drift_score")` calls would return `None`, silently disabling the drift score conditioning in execution cost estimation.

**Downstream impact:** The two ShockVector construction paths produce structurally different objects. The backtester relies on `compute_shock_vectors` which provides `drift_score` and `systemic_stress`. The `detect_with_shock_context` path (used for single-bar API calls) produces neither. Consumers must know which construction path was used, but nothing in the ShockVector schema distinguishes them.

**Recommended fix:** Unify structural feature computation. Either:
1. Add `drift_score` and `systemic_stress` computation to `detect_with_shock_context`, or
2. Create a shared helper that both methods call, producing a consistent feature set.

---

### F-03 — `REGIME_JUMP_MODE_LOSS_WEIGHT` imported but never used [P1 HIGH]

**Invariant at risk:** Config-code alignment — unused config suggests incomplete integration.

**Proof:**
- config.py:249 defines `REGIME_JUMP_MODE_LOSS_WEIGHT = 0.1` (a float) with status ACTIVE.
- jump_model_pypi.py:91 imports `REGIME_JUMP_MODE_LOSS_WEIGHT` inside `__init__`.
- jump_model_pypi.py:81 has `mode_loss: bool = True` as a constructor parameter (a boolean, not a float).
- jump_model_pypi.py:103 stores `self.mode_loss = mode_loss` (the boolean True, NOT the config float 0.1).
- jump_model_pypi.py:167 passes `mode_loss=self.mode_loss` to the PyPI `JumpModel`.
- The config constant `REGIME_JUMP_MODE_LOSS_WEIGHT` (float 0.1) is **never referenced** after import.

**Downstream impact:** The config constant exists to control the mode loss penalty weight for continuous jump models, but the actual weight value is never applied. The PyPI JumpModel receives only a boolean `mode_loss=True`, meaning the default internal weight of the library is used rather than the configured value. This may represent an incomplete integration where the weight was intended to be passed as a separate parameter.

**Recommended fix:** Either:
1. Wire the config value into the JumpModel call if the PyPI package supports a `mode_loss_weight` parameter, or
2. Remove the import and mark the config constant as DEPRECATED if the PyPI package only accepts a boolean toggle.

---

### F-04 — `compute_shock_vectors` hardcodes `ensemble_model_type="hmm"` for all bars [P2 MEDIUM]

**Proof:**
- shock_vector.py:487: `ensemble_model_type="hmm"` is hardcoded in every ShockVector constructed by `compute_shock_vectors`.
- In contrast, `detect_with_shock_context` (detector.py:805) correctly sets `ensemble_model_type=regime_out.model_type`, which reflects the actual model used (could be "hmm", "jump", "rule", or "ensemble").
- When the system uses ensemble detection, the batch ShockVectors incorrectly claim "hmm" as the model type.

**Downstream impact:** Informational field — no downstream logic currently branches on `ensemble_model_type`. But it creates a diagnostic/debugging inaccuracy and would become a correctness issue if any consumer starts conditioning on this field.

**Recommended fix:** Pass the actual model type from the regime detection output into `compute_shock_vectors`, or accept an optional `model_type` parameter.

---

### F-05 — HMM `_apply_min_duration` confidence tie-breaking logic is no-op [P2 MEDIUM]

**Proof:**
- detector.py:166-170: When `left_state` and `right_state` are both available, the method computes `left_score` and `right_score`. However:
  - `left_score = float(conf[i:j].mean()) if left_state == int(vals[max(0, i-1)]) else 0.0`
  - The condition `left_state == int(vals[max(0, i-1)])` is checking if the left neighbor's state equals the leftmost value of the short run's **already-replaced** left neighbor. Since `left_state = int(vals[i-1])` and we're comparing `left_state == int(vals[max(0, i-1)])`, this is always True (they're the same index when i > 0).
  - Similarly, `right_score = float(conf[i:j].mean()) if right_state == int(vals[min(j, n-1)]) else 0.0` is always True when j < n.
  - Since both scores use the same `conf[i:j].mean()` value, they will always be equal, and the tie-break always picks `left_state` (the "prefer left" logic at line 170).

**Downstream impact:** The confidence-based tie-breaking never actually selects the right neighbor — it always selects left. This means short regime runs are always merged leftward rather than toward the higher-confidence neighbor. The behavioral impact is mild since min_duration smoothing is a secondary correction, but it doesn't match the documented intent.

**Recommended fix:** Fix the scoring logic to compare the confidence of the neighboring segments rather than the short segment itself.

---

### F-06 — `detect_full` ignores explicit `method="jump"` when `REGIME_ENSEMBLE_ENABLED=True` [P2 MEDIUM]

**Proof:**
- detector.py:561-569:
  ```python
  def detect_full(self, features):
      if REGIME_ENSEMBLE_ENABLED and self.method == "hmm":
          return self.detect_ensemble(features)
      if self.method == "hmm":
          return self._hmm_detect(features)
      if self.method == "jump":
          return self._jump_detect(features)
      return self._rule_detect(features)
  ```
- When a user creates `RegimeDetector(method="jump")`, the jump model is used correctly.
- When a user creates `RegimeDetector(method="ensemble")`, `detect_full` falls through all conditions and returns `self._rule_detect(features)` — the ensemble method is never called.
- The ensemble is only triggered when `REGIME_ENSEMBLE_ENABLED=True` AND `method="hmm"`. There is no way to explicitly request ensemble detection via the `method` parameter.

**Downstream impact:** The `method` parameter accepts "ensemble" as a value (no validation), but it silently falls through to rule-based detection. This is a configuration trap — a user setting `REGIME_MODEL_TYPE="ensemble"` in config would get rule-based detection instead.

**Recommended fix:** Add `method="ensemble"` handling:
```python
if self.method == "ensemble" or (REGIME_ENSEMBLE_ENABLED and self.method == "hmm"):
    return self.detect_ensemble(features)
```

---

### F-07 — `map_raw_states_to_regimes` uses heuristic mapping that can be unstable across refits [P2 MEDIUM]

**Proof:**
- hmm.py:612-660: `map_raw_states_to_regimes` assigns semantic labels by computing per-state statistics (mean return, mean NATR, mean SMA slope, mean Hurst) and picking the "most bull" state as regime 0, "most bear" as regime 1, etc.
- This heuristic depends on the feature distribution within each HMM state. When the HMM is refitted on new data, the same market conditions can produce different raw state assignments (label permutation problem inherent to EM), and the heuristic mapping may flip labels.
- detector.py:638-711 provides `map_raw_states_to_regimes_stable` which uses Wasserstein distance matching to reference distributions — specifically designed to solve this problem. However, `_hmm_detect` (detector.py:280) calls `map_raw_states_to_regimes` (the unstable version), NOT the stable variant.

**Downstream impact:** Regime labels may flip between HMM refits, causing downstream consumers (position sizing, trade gating, covariance selection) to suddenly change behavior. The online update path (`OnlineRegimeUpdater`) inherits whatever mapping was established at the last full fit.

**Recommended fix:** Switch `_hmm_detect` and `_jump_detect` to use `map_raw_states_to_regimes_stable` instead of the heuristic version.

---

### F-08 — `OnlineRegimeUpdater` does not validate HMM state count matches expected 4 regimes [P2 MEDIUM]

**Proof:**
- online_update.py:36-65: `OnlineRegimeUpdater.__init__` accepts any `GaussianHMM` instance. It caches `self._trans_log` from the model's transition matrix but does not verify `hmm_model.n_states == 4`.
- online_update.py:143-177: `update_regime_for_security` returns `regime = int(np.argmax(state_prob))`. If the HMM has 6 states (via BIC auto-selection), regime values 4 and 5 will be produced, which are outside the canonical [0, 3] range and would fail ShockVector validation.

**Downstream impact:** When `REGIME_HMM_AUTO_SELECT_STATES=True` and BIC selects more than 4 states, the online updater can produce regime IDs outside [0, 3]. These would be rejected by ShockVector's `__post_init__` validation (`ValueError`) if they reach ShockVector construction.

**Recommended fix:** Add state count validation in `OnlineRegimeUpdater.__init__` or apply `map_raw_states_to_regimes` inside `update_regime_for_security`.

---

### F-09 — `compute_shock_vectors` per-bar systemic stress loop is O(n^2) [P3 LOW]

**Proof:**
- shock_vector.py:457-461: Computing the expanding percentile for each bar requires iterating over all prior bars. For a 2500-bar series (10 years), this is ~3.1M comparisons per ticker.
- With 500 tickers in the universe, total work is ~1.5B comparisons just for systemic stress.

**Downstream impact:** Performance concern for large backtests. No correctness issue.

**Recommended fix:** Use `pd.Series.expanding().rank()` or numpy vectorized percentile.

---

### F-10 — `ShockVectorValidator.validate` does not check `timestamp` type or value [P3 LOW]

**Proof:**
- shock_vector.py:176-281: The validator checks schema_version, hmm_regime, hmm_confidence, hmm_uncertainty, bocpd_*, ticker, structural_features, transition_matrix, and ensemble_model_type. It does NOT validate:
  - `timestamp` is a `datetime` instance
  - `jump_detected` is a `bool`
  - `jump_magnitude` is numeric

**Downstream impact:** A ShockVector with `timestamp="not_a_datetime"` or `jump_detected=42` would pass validation. Minor because `__post_init__` provides some defense, but the validator's purpose is comprehensive post-hoc checking.

---

### F-11 — `RegimeConsensus._consensus_history` unbounded growth [P3 LOW]

**Proof:**
- consensus.py:83-136: `compute_consensus` appends to `self._consensus_history` on every call. `detect_divergence` reads the last `window` entries. But `_consensus_history` is never trimmed — over a long-running session (e.g., autopilot daily iterations), it grows without bound.
- `reset_history` (line 271) exists but must be called explicitly.

**Downstream impact:** Memory growth proportional to number of `compute_consensus` calls. Negligible for typical usage (hundreds of calls), but could matter for long-running processes.

---

### F-12 — `GaussianHMM._smooth_duration` modifies state labels but not probabilities [P3 LOW]

**Proof:**
- hmm.py:257-290: `_smooth_duration` changes `states[i:j]` for short runs but does not update the corresponding rows in `state_probs`. The returned states and the probabilities used by `fit()` (line 335) can be inconsistent: the label says regime X but the posterior probability might peak at regime Y.
- detector.py:297 computes `confidence = probs.max(axis=1)` from the raw (unsmoothed) probabilities, while the regime label comes from the smoothed states.

**Downstream impact:** Minor inconsistency — confidence might not accurately reflect the smoothed regime label for bars where duration smoothing changed the assignment.

---

### F-13 — `CorrelationRegimeDetector` threshold semantics differ from other regime IDs [P3 LOW]

**Proof:**
- correlation.py:123-147: `detect_correlation_spike` produces a binary {0, 1} series named `corr_regime`, where 1 = correlation spike (stress) and 0 = normal.
- This differs from the canonical 4-regime mapping (0-3) used everywhere else. The `corr_regime` output is used as a feature column, not as a regime label, so no semantic conflict exists in practice.
- However, the naming `corr_regime` could confuse auditors or developers into thinking it follows the canonical regime ID convention.

**Downstream impact:** None — used as a feature column by `features/pipeline.py`, not as a regime label. Naming could cause confusion.

---

## Config Constant Audit (T5)

### All Config Constants Referenced by Regime Subsystem

| Constant | File(s) | Import Type | Verified in config.py | Value |
|---|---|---|---|---|
| `REGIME_MODEL_TYPE` | detector.py:25 | top_level | Yes (line 224) | `_cfg.regime.model_type` |
| `REGIME_HMM_STATES` | detector.py:26 | top_level | Yes (line 225) | `_cfg.regime.n_states` |
| `REGIME_HMM_MAX_ITER` | detector.py:27 | top_level | Yes (line 226) | `_cfg.regime.hmm_max_iter` |
| `REGIME_HMM_STICKINESS` | detector.py:28 | top_level | Yes (line 227) | `_cfg.regime.hmm_stickiness` |
| `REGIME_MIN_DURATION` | detector.py:29 | top_level | Yes (line 228) | `_cfg.regime.min_duration` |
| `REGIME_HMM_AUTO_SELECT_STATES` | detector.py:30 | top_level | Yes (line 232) | `_cfg.regime.hmm_auto_select_states` |
| `REGIME_HMM_MIN_STATES` | detector.py:31 | top_level | Yes (line 233) | `_cfg.regime.hmm_min_states` |
| `REGIME_HMM_MAX_STATES` | detector.py:32 | top_level | Yes (line 234) | `_cfg.regime.hmm_max_states` |
| `REGIME_JUMP_MODEL_ENABLED` | detector.py:33 | top_level | Yes (line 235) | `_cfg.regime.jump_model_enabled` |
| `REGIME_JUMP_PENALTY` | detector.py:34 | top_level | Yes (line 236) | `_cfg.regime.jump_penalty` |
| `REGIME_EXPECTED_CHANGES_PER_YEAR` | detector.py:35 | top_level | Yes (line 237) | `_cfg.regime.expected_changes_per_year` |
| `REGIME_ENSEMBLE_ENABLED` | detector.py:36 | top_level | Yes (line 238) | `_cfg.regime.ensemble_enabled` |
| `REGIME_ENSEMBLE_CONSENSUS_THRESHOLD` | detector.py:37 | top_level | Yes (line 239) | `_cfg.regime.ensemble_consensus_threshold` |
| `BOCPD_ENABLED` | detector.py:38 | top_level | Yes (line 288) | `True` |
| `BOCPD_HAZARD_LAMBDA` | detector.py:39 | top_level | Yes (line 290) | `1.0 / 60` |
| `BOCPD_HAZARD_FUNCTION` | detector.py:40 | top_level | Yes (line 289) | `"constant"` |
| `BOCPD_RUNLENGTH_DEPTH` | detector.py:41 | top_level | Yes (line 292) | `200` |
| `BOCPD_CHANGEPOINT_THRESHOLD` | detector.py:42 | top_level | Yes (line 293) | `0.50` |
| `SHOCK_VECTOR_SCHEMA_VERSION` | detector.py:43 | top_level | Yes (line 296) | `"1.0"` |
| `SHOCK_VECTOR_INCLUDE_STRUCTURAL` | detector.py:44 | top_level | Yes (line 297) | `True` |
| `REGIME_HMM_PRIOR_WEIGHT` | detector.py:240 | lazy | Yes (line 230) | `_cfg.regime.hmm_prior_weight` |
| `REGIME_HMM_COVARIANCE_TYPE` | detector.py:240 | lazy | Yes (line 231) | `_cfg.regime.hmm_covariance_type` |
| `REGIME_JUMP_USE_PYPI_PACKAGE` | detector.py:324 | lazy | Yes (line 242) | `True` |
| `REGIME_ENSEMBLE_DEFAULT_WEIGHTS` | detector.py:402 | lazy | Yes (line 253) | `{"hmm":0.5, "rule":0.3, "jump":0.2}` |
| `REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD` | detector.py:404 | lazy | Yes (line 258) | `0.40` |
| `REGIME_ENSEMBLE_UNCERTAIN_FALLBACK` | detector.py:405 | lazy | Yes (line 259) | `3` (high_volatility) |
| `REGIME_NAMES` | detector.py:933 | conditional | Yes (line 217) | `{0: "trending_bull", ...}` |
| `REGIME_EXPANDED_FEATURES_ENABLED` | hmm.py:560 | lazy | Yes (line 282) | `True` |
| `REGIME_JUMP_CV_FOLDS` | jump_model_pypi.py:85 | lazy | Yes (line 243) | `5` |
| `REGIME_JUMP_LAMBDA_RANGE` | jump_model_pypi.py:86 | lazy | Yes (line 244) | `(0.005, 0.15)` |
| `REGIME_JUMP_LAMBDA_STEPS` | jump_model_pypi.py:87 | lazy | Yes (line 245) | `20` |
| `REGIME_JUMP_MAX_ITER` | jump_model_pypi.py:88 | lazy | Yes (line 246) | `50` |
| `REGIME_JUMP_TOL` | jump_model_pypi.py:89 | lazy | Yes (line 247) | `1e-6` |
| `REGIME_JUMP_USE_CONTINUOUS` | jump_model_pypi.py:90 | lazy | Yes (line 248) | `True` |
| `REGIME_JUMP_MODE_LOSS_WEIGHT` | jump_model_pypi.py:91 | lazy | Yes (line 249) | `0.1` (**UNUSED — F-03**) |
| `REGIME_JUMP_PENALTY` | jump_model_pypi.py:314 | conditional | Yes (line 236) | `_cfg.regime.jump_penalty` |
| `REGIME_CONSENSUS_THRESHOLD` | consensus.py:55 | lazy | Yes (line 272) | `0.80` |
| `REGIME_CONSENSUS_EARLY_WARNING` | consensus.py:55 | lazy | Yes (line 273) | `0.60` |
| `REGIME_CONSENSUS_DIVERGENCE_WINDOW` | consensus.py:55 | lazy | Yes (line 274) | `20` |
| `REGIME_CONSENSUS_DIVERGENCE_SLOPE` | consensus.py:55 | lazy | Yes (line 275) | `-0.01` |
| `REGIME_ONLINE_REFIT_DAYS` | online_update.py:60 | lazy | Yes (line 279) | `30` |
| `REGIME_UNCERTAINTY_ENTROPY_THRESHOLD` | uncertainty_gate.py:53 | lazy | Yes (line 262) | `0.50` |
| `REGIME_UNCERTAINTY_STRESS_THRESHOLD` | uncertainty_gate.py:54 | lazy | Yes (line 263) | `0.80` |
| `REGIME_UNCERTAINTY_SIZING_MAP` | uncertainty_gate.py:55 | lazy | Yes (line 264) | `{0.0:1.0, 0.5:0.95, 1.0:0.85}` |
| `REGIME_UNCERTAINTY_MIN_MULTIPLIER` | uncertainty_gate.py:56 | lazy | Yes (line 269) | `0.80` |

**Result: 44 config constants verified.** All exist in config.py with correct types and ACTIVE status. 1 unused import identified (F-03).

### Lazy Import Safety

All lazy/conditional config imports are inside method bodies or `__init__`, never at module level (except the 20 top-level imports in detector.py:24-44). This ensures:
- No circular import risk from config → regime path
- Import failures surface at call time with clear tracebacks
- Optional features (BOCPD, PyPI jump model) gracefully degrade

---

## Dependency Edge Verification (T5)

### Inbound Edges (17 edges from 7 consumer modules)

| Consumer | Line | Symbol | Import Type | Verified |
|---|---|---|---|---|
| backtest/engine.py | 77 | `compute_shock_vectors, ShockVector` | top_level | Yes |
| backtest/engine.py | 78 | `UncertaintyGate` | top_level | Yes |
| backtest/execution.py | 29 | `ShockVector` | TYPE_CHECKING | Yes |
| risk/position_sizer.py | 27 | `UncertaintyGate` | top_level | Yes |
| autopilot/engine.py | 60 | `RegimeDetector` | top_level | Yes |
| autopilot/engine.py | 61 | `UncertaintyGate` | top_level | Yes |
| features/pipeline.py | 1303 | `CorrelationRegimeDetector` | conditional | Yes |
| api/orchestrator.py | 45, 225, 292 | `RegimeDetector` | lazy | Yes |
| api/services/data_helpers.py | 491 | `RegimeDetector` | conditional | Yes |
| run_backtest.py | 38 | `RegimeDetector` | top_level | Yes |
| run_predict.py | 29 | `RegimeDetector` | top_level | Yes |
| run_retrain.py | 29 | `RegimeDetector` | top_level | Yes |
| run_train.py | 29 | `RegimeDetector` | top_level | Yes |
| scripts/compare_regime_models.py | 30, 31 | `RegimeDetector`, `build_hmm_observation_matrix` | top_level | Yes |

### Outbound Edges (11 edges to config only)

All regime → config edges verified against DEPENDENCY_EDGES.json. No undocumented cross-module dependencies found.

### Internal Edges (21 within regime/)

All internal imports verified. Key architecture:
- `__init__.py` re-exports 21 symbols from 11 submodules
- `detector.py` is the orchestrator importing from `hmm.py`, `jump_model_legacy.py`, `confidence_calibrator.py`, `bocpd.py` (conditional), `jump_model_pypi.py` (lazy), `shock_vector.py` (lazy)
- `hmm.py` imports from `bocpd.py` (conditional, for structural features)
- `jump_model_pypi.py` imports from `jump_model_legacy.py` (JumpModelResult)
- `jump_model.py` re-exports from `jump_model_legacy.py` (backward compat shim)
- `shock_vector.py` imports from `bocpd.py` (conditional, in compute_shock_vectors)

### Optional Package Dependencies

| Package | File | Import Type | Fallback |
|---|---|---|---|
| `scipy.special.gammaln` | bocpd.py:6 | top_level | None (hard requirement for BOCPD) |
| `jumpmodels.jump.JumpModel` | jump_model_pypi.py:128 | lazy (inside `fit`) | Falls back to legacy StatisticalJumpModel via `ImportError` catch |
| `sklearn.cluster.KMeans` | jump_model_legacy.py:135 | conditional (inside `_kmeans_init`) | Falls back to evenly-spaced init |

---

## Test Coverage Assessment

| Test File | Coverage Target |
|---|---|
| `test_regime_integration.py` | End-to-end regime detection pipeline |
| `test_regime_detection_integration.py` | Multi-engine detection |
| `test_regime_payload.py` | ShockVector payload structure |
| `test_regime_consensus.py` | Consensus module |
| `test_regime_uncertainty.py` | Uncertainty gating |
| `test_regime_covariance_wiring.py` | Regime-conditional covariance |
| `test_regime_capacity.py` | Capacity testing |
| `test_shock_vector.py` | ShockVector unit tests |
| `test_shock_mode_policy.py` | ShockModePolicy integration |
| `test_uncertainty_gate_wiring.py` | UncertaintyGate consumer wiring |
| `test_spec_p01_regime_stats.py` | Regime statistics |

**Coverage gaps identified:**
- No dedicated test for `map_raw_states_to_regimes` label stability across refits
- No dedicated test for `from_dict` with unknown keys (forward compatibility)
- No test for `compute_shock_vectors` vs `detect_with_shock_context` structural feature parity
- No test for `OnlineRegimeUpdater` with >4 HMM states
- No test for `detect_full` with `method="ensemble"` when `REGIME_ENSEMBLE_ENABLED=False`
- `bocpd.py`, `confidence_calibrator.py`, and `jump_model_legacy.py` have no dedicated test files (tested indirectly through integration tests)

---

## Findings Summary

| ID | Severity | Category | Component | Description |
|---|---|---|---|---|
| F-01 | P1 HIGH | Contract | shock_vector.py | `from_dict` rejects unknown keys, breaking documented forward compatibility |
| F-02 | P1 HIGH | Contract | shock_vector.py + detector.py | Two ShockVector construction paths produce different structural features |
| F-03 | P1 HIGH | Config | jump_model_pypi.py | `REGIME_JUMP_MODE_LOSS_WEIGHT` imported but never used — dead integration |
| F-04 | P2 MEDIUM | Correctness | shock_vector.py | `compute_shock_vectors` hardcodes `ensemble_model_type="hmm"` |
| F-05 | P2 MEDIUM | Correctness | detector.py | `_apply_min_duration` confidence tie-breaking is effectively a no-op |
| F-06 | P2 MEDIUM | Correctness | detector.py | `detect_full` ignores `method="ensemble"`, falls through to rule-based |
| F-07 | P2 MEDIUM | Correctness | detector.py + hmm.py | Unstable state mapping used instead of available Wasserstein-based stable mapping |
| F-08 | P2 MEDIUM | Correctness | online_update.py | No validation that HMM state count == 4, can produce out-of-range regime IDs |
| F-09 | P3 LOW | Performance | shock_vector.py | O(n^2) systemic stress computation |
| F-10 | P3 LOW | Correctness | shock_vector.py | Validator missing checks for timestamp, jump_detected, jump_magnitude |
| F-11 | P3 LOW | Performance | consensus.py | Unbounded consensus history growth |
| F-12 | P3 LOW | Correctness | hmm.py | Duration smoothing modifies labels but not probabilities |
| F-13 | P3 LOW | Naming | correlation.py | `corr_regime` binary naming may confuse with canonical 4-regime IDs |

---

## Contract Verdicts

### ShockVector Contract: **COMPATIBLE with caveats**

The ShockVector schema is well-designed and properly version-locked. The 13-field structure, validation, and serialization are correct for the current V1 schema. The two construction paths (`compute_shock_vectors` and `detect_with_shock_context`) both produce valid ShockVectors but with different structural feature sets (F-02). Forward compatibility is claimed but not implemented (F-01). No breaking changes detected against current consumers.

### UncertaintyGate Contract: **FULLY COMPATIBLE**

All 3 consumers instantiate `UncertaintyGate()` with identical default config parameters. The sizing map interpolation is correct. The multiplier range [0.80, 1.0] is reasonable — at maximum uncertainty, positions are reduced by only 20%. The stress threshold (0.80) and entropy threshold (0.50) are consistent across all paths.

### RegimeDetector Output Contract: **COMPATIBLE**

Regime labels are consistently in [0, 3] across all detection engines. Confidence values are in [0, 1]. Probability columns follow the `regime_prob_N` naming convention. The `regime_features()` method produces a complete feature set consumed by autopilot and entry points. The unstable state mapping (F-07) is a correctness concern but does not break the contract.

---

## Recommended Follow-Up Actions

### Must-fix before next backtest run
1. **F-01**: Add unknown-key filtering to `ShockVector.from_dict` (prevents future breakage)
2. **F-02**: Unify structural feature computation between the two ShockVector construction paths

### Should-fix in next maintenance cycle
3. **F-03**: Resolve `REGIME_JUMP_MODE_LOSS_WEIGHT` — wire it or remove it
4. **F-06**: Add `method="ensemble"` handling in `detect_full`
5. **F-07**: Switch to `map_raw_states_to_regimes_stable` in production detection paths
6. **F-08**: Add state count validation in `OnlineRegimeUpdater`

### Nice-to-have
7. **F-04**: Pass actual model type to `compute_shock_vectors`
8. **F-05**: Fix `_apply_min_duration` tie-breaking logic
9. **F-09**: Vectorize systemic stress computation
10. **F-10**: Add missing validator checks
11. Add dedicated tests for: `from_dict` forward compat, structural feature parity, >4 state online update, ensemble method dispatch

---

## Transition Notes for Subsystem 5 Audit (Backtesting + Risk)

### Carry Forward
- ShockVector schema (13 fields, version "1.0") — consumers access `hmm_uncertainty`, `bocpd_changepoint_prob`, `structural_features["drift_score"]`, `structural_features["systemic_stress"]`, `is_shock_event()`.
- UncertaintyGate produces multipliers in [0.80, 1.0] from config defaults. All consumers use identical default instantiation.
- Regime labels are integers 0-3. `risk/position_sizer.py` converts to string names (`"trending_bull"`, etc.) for regime stats keying.
- **F-02 is relevant**: `backtest/engine.py` calls `compute_shock_vectors` which produces `drift_score` and `systemic_stress`. The alternative path (`detect_with_shock_context`) does NOT produce these. Verify the backtester always uses the correct path.

### Boundary Checks for Subsystem 5
- `backtest_to_regime_shock_1`: Verify all ShockVector field accesses in `backtest/engine.py` and `backtest/execution.py` use `.get()` for `structural_features` (they do — confirmed in this audit).
- `backtest_to_regime_uncertainty_2`: Verify `UncertaintyGate` instantiation in `backtest/engine.py:278` and `risk/position_sizer.py:105` use no overrides (they don't — confirmed).
