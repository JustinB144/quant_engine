# Feature Spec: Health Check System — Complete Rewrite of All 15 Checks

> **Status:** Approved
> **Author:** justin
> **Date:** 2026-02-23
> **Estimated effort:** ~14 hours across 9 tasks

---

## Why

Every health check in the system is fundamentally flawed. The checks validate config existence rather than actual functionality, use fallback scores of 50.0 that inflate results when data is missing, use arbitrary thresholds with no statistical basis, and measure the wrong metrics. Examples: (1) `_check_signal_decay` measures prediction error autocorrelation but the real risk is alpha decay over time — it should track rolling Sharpe or IC. (2) `_check_correlation_regime` reads random parquet files from the cache directory instead of analyzing actual portfolio holdings. (3) `_check_execution_quality` uses paper trading's simulated slippage to score real execution — it's measuring fake numbers. (4) `_check_survivorship_bias` gives 90 points just for having WRDS enabled, regardless of whether survivorship-free universe construction is actually running. (5) All checks return `score=50.0` on exception, making failures look "average" instead of flagging them.

## What

Rewrite every health check to measure real, observable system behavior instead of config/code existence. Each check should: (a) measure something that actually matters for trading performance, (b) have a clear pass/warn/fail threshold based on quantitative reasoning, (c) return `status=UNAVAILABLE` when data is missing (not score=50), and (d) include an explanation of what the score means and what the user should do if it's low.

## Constraints

### Must-haves
- All 15 existing checks rewritten with correct metrics
- Exception handlers return `UNAVAILABLE` status, not `score=50`
- Each check includes `methodology` field explaining what it measures
- Thresholds documented with rationale
- Domain weights justified (current: Data=25%, Signal=25%, Risk=20%, Execution=15%, Governance=15%)

### Must-nots
- Do NOT add checks that require live market data (system may be offline)
- Do NOT make health checks take >5 seconds total
- Do NOT remove any existing check names (API consumers depend on them)

## Tasks

### T1: Create HealthCheckResult dataclass and rewrite exception handling

**What:** Replace dict-based check results with a structured dataclass, and make all exception handlers return `UNAVAILABLE` instead of `score=50`.

**Files:**
- `api/services/health_service.py` — add dataclass, update all `except` blocks

**Implementation notes:**
```python
@dataclass
class HealthCheckResult:
    name: str
    domain: str  # 'data', 'signal', 'risk', 'execution', 'governance'
    score: float  # 0-100
    status: str  # 'PASS', 'WARN', 'FAIL', 'UNAVAILABLE'
    explanation: str  # Human-readable: what the score means
    methodology: str  # What this check measures and why
    data_available: bool  # Whether underlying data existed
    raw_metrics: Dict  # Actual measured values
    thresholds: Dict  # What thresholds were used

# Exception pattern for ALL checks:
try:
    # ... actual check logic ...
except Exception as e:
    logger.warning("Health check '%s' failed: %s", name, e, exc_info=True)
    return HealthCheckResult(
        name=name,
        domain=domain,
        score=0,  # NOT 50
        status="UNAVAILABLE",
        explanation=f"Check could not run: {e}",
        methodology="N/A — check failed to execute",
        data_available=False,
        raw_metrics={},
        thresholds={},
    )
```

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService, HealthCheckResult
svc = HealthService()
# Force a check to fail by passing bad data
# It should return UNAVAILABLE, not score=50
print('HealthCheckResult dataclass exists')
"
```

---

### T2: Rewrite signal quality checks (decay, distribution, ensemble)

**What:** Fix `_check_signal_decay`, `_check_prediction_distribution`, and `_check_ensemble_disagreement` to measure real signal quality.

**Files:**
- `api/services/health_service.py` — rewrite 3 methods

**Implementation notes:**

**Signal Decay (was: prediction error autocorrelation):**
- New metric: Rolling IC (information coefficient) trend over last N prediction batches
- If IC is declining monotonically → decay detected
- Score: IC_recent / IC_initial — ratio > 0.8 = PASS (85), > 0.5 = WARN (55), else FAIL (25)
- If no prediction history: UNAVAILABLE
- Methodology: "Measures whether the model's predictive power (rank correlation between predicted and realized returns) is declining over time."

**Prediction Distribution:**
- Current correctly checks variance, but should also check for prediction clustering
- Add: check that prediction quantiles are spread across the range, not clustered at 0
- Score: std > 0.005 AND IQR > 0.003 = PASS (85), std > 0.001 = WARN (55), else FAIL (25)
- Methodology: "Checks that predictions have meaningful dispersion, indicating the model distinguishes between stocks rather than predicting near-zero for all."

**Ensemble Disagreement:**
- Current checks regime model spread, but doesn't check if disagreement is informative
- New: measure correlation between ensemble disagreement and realized volatility
- High disagreement should correlate with high vol (uncertainty = risk)
- Score: correlation > 0.3 = PASS (85), > 0.1 = WARN (55), else FAIL (25)
- Methodology: "Checks that model uncertainty (disagreement between regime-specific models) is correlated with realized market uncertainty."

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
# These should all return HealthCheckResult with methodology field
print('Signal quality checks rewritten')
"
```

---

### T3: Rewrite data quality checks (survivorship, anomalies, microstructure)

**What:** Fix `_check_survivorship_bias`, `_check_data_quality_anomalies`, and `_check_market_microstructure` to measure real data quality.

**Files:**
- `api/services/health_service.py` — rewrite 3 methods

**Implementation notes:**

**Survivorship Bias:**
- Current: gives 90 points just for WRDS being enabled
- New: check if universe construction actually uses delisting returns
  1. Verify DATA_CACHE_DIR has parquets with `total_ret` column (not just `Return`)
  2. Count tickers that have been delisted (last bar date < today - 1 year)
  3. Score: has_total_ret AND delisted_count > 0 = PASS (90), has_total_ret = WARN (60), neither = FAIL (20)
- Methodology: "Checks that the data actually includes delisted securities and uses total return (including delisting returns), not just price return."

**Data Quality Anomalies:**
- Current: checks volume zeros and extreme returns on random files
- New: systematic scan across cached universe
  1. For each ticker: check zero-volume fraction, extreme daily returns, stale data (no update in 30 days)
  2. Aggregate: % of tickers with quality issues
  3. Score: <5% tickers with issues = PASS (85), <20% = WARN (55), else FAIL (25)
- Methodology: "Systematically scans all cached data for quality issues: zero-volume bars, extreme returns (>40% daily), and stale data."

**Market Microstructure:**
- Current: checks dollar volume >= $1M
- New: check if universe has sufficient liquidity for the position sizes being used
  1. Compute median daily dollar volume across universe
  2. Compare to BACKTEST_ASSUMED_CAPITAL * POSITION_SIZE_PCT
  3. If position_notional > 1% of median_volume → liquidity concern
  4. Score based on participation rate: <0.5% = PASS (85), <2% = WARN (55), else FAIL (25)
- Methodology: "Checks that the backtest's assumed capital and position sizes are realistic given the liquidity of the traded universe."

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
print('Data quality checks rewritten')
"
```

---

### T4: Rewrite risk management checks (tail risk, correlation, capital utilization)

**What:** Fix `_check_tail_risk`, `_check_correlation_regime`, and `_check_capital_utilization` to measure real risk.

**Files:**
- `api/services/health_service.py` — rewrite 3 methods

**Implementation notes:**

**Tail Risk:**
- Current: checks CVaR5 and tail ratio on paper trading returns
- Issue: paper trading returns are simulated, not real
- New: compute from backtest daily equity curve (which exists if any backtest has been run)
  1. CVaR at 95%: expected loss on worst 5% of days
  2. Max drawdown recovery time
  3. Score: CVaR > -3% AND recovery < 60 days = PASS (85)
- Methodology: "Measures tail risk from the most recent backtest: CVaR (expected shortfall) at 95% and maximum drawdown recovery time."

**Correlation/Regime:**
- Current: reads random parquet files and computes pairwise correlation
- New: compute correlation of ACTUAL portfolio holdings (if any exist in paper trader state)
  1. Load paper trader state to get current positions
  2. Compute pairwise correlation of held tickers
  3. Score based on average pairwise correlation: <0.4 = PASS (85), <0.65 = WARN (55), else FAIL (25)
- If no positions: check recent backtest trades' correlation
- Methodology: "Measures portfolio concentration risk by computing pairwise correlation of current or recent holdings."

**Capital Utilization:**
- Current logic is reasonable but should also check cash drag
- New: add check for concentrated vs diversified allocation
  1. Compute Herfindahl index of position sizes
  2. Low HHI (diversified) + moderate utilization = PASS
  3. High HHI (concentrated) regardless of utilization = WARN
- Methodology: "Measures how efficiently capital is deployed and whether allocation is diversified across positions."

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
print('Risk checks rewritten')
"
```

---

### T5: Rewrite execution quality check

**What:** Fix `_check_execution_quality` to measure actual execution metrics, not simulated ones.

**Files:**
- `api/services/health_service.py` — rewrite `_check_execution_quality()`

**Implementation notes:**
- Current bug: uses paper trading's simulated slippage (which is just tx_cost_bps applied uniformly)
- New: analyze backtest TCA report if available
  1. Check that predicted impact correlates with realized slippage (execution model accuracy)
  2. Check that median slippage is within expected range (< 2x spread)
  3. Check fill ratio distribution (should be > 0.8 for liquid stocks)
  4. If no TCA data: UNAVAILABLE
- Score: correlation > 0.3 AND median_slip < 10bps = PASS (85)
- Methodology: "Validates execution model accuracy by comparing predicted market impact to realized slippage from backtests."

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
print('Execution quality check rewritten')
"
```

---

### T6: Rewrite model governance checks (CV gap, retraining, information ratio)

**What:** Fix `_check_cv_gap_trend`, `_check_retraining_effectiveness`, `_check_information_ratio`, and `_check_feature_importance_drift`.

**Files:**
- `api/services/health_service.py` — rewrite 4 methods

**Implementation notes:**

**CV Gap Trend:**
- Current: reads model metadata, checks if IS-OOS gap is increasing
- Improvement: also check absolute gap level, not just trend
- Gap > 0.15 is always FAIL regardless of trend (even stable overfitting is bad)
- Score: gap < 0.05 AND trend < 0 = PASS (85), gap < 0.10 = WARN (55), else FAIL (25)
- Methodology: "Tracks the gap between in-sample and out-of-sample model performance across training runs. A growing gap indicates increasing overfitting."

**Retraining Effectiveness:**
- Current: compares latest holdout Spearman to average
- Improvement: track direction — is model getting better or worse over time?
- Score based on slope of holdout performance across versions
- Score: improving = PASS (85), stable = WARN (65), declining = FAIL (25)
- Methodology: "Measures whether model quality is improving, stable, or degrading across retraining cycles."

**Information Ratio:**
- Rename to `Signal Profitability` for clarity
- Check long/short signal returns, not just IC
- Score: long_return > 0 AND IC > 0 = PASS, long_return > 0 = WARN, else FAIL
- Methodology: "Measures whether model predictions translate into profitable trades: positive long signal returns and positive information coefficient."

**Feature Importance Drift:**
- Current: Spearman correlation of feature rankings (reasonable)
- Improvement: also identify which features changed most
- Add explanation: "Top 3 features with largest rank change: feature_A (+15), feature_B (-12), feature_C (+8)"
- Methodology: "Detects if the model's reliance on features is shifting, which could indicate regime change or data drift."

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
print('Governance checks rewritten')
"
```

---

### T7: Rewrite regime transition health check

**What:** Fix `_check_regime_transition_health` to properly assess regime detection quality.

**Files:**
- `api/services/health_service.py` — rewrite method

**Implementation notes:**
- Current: checks HMM transition matrix diagonal > 0.95 (regime stickiness)
- Issue: regime stickiness depends on min_duration parameter, not just HMM quality
- New checks:
  1. Regime entropy: are we stuck in one regime? (low entropy = bad)
  2. Regime prediction vs realized: do regimes predict volatility changes?
  3. Transition frequency: too many transitions = noisy, too few = insensitive
- Score: entropy > 0.5 AND predictive_corr > 0.2 = PASS (85)
- Methodology: "Assesses whether regime detection produces meaningful, predictive regime labels by checking regime diversity, predictive power, and transition stability."

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
print('Regime transition check rewritten')
"
```

---

### T8: Update domain scoring and overall health computation

**What:** Update `compute_comprehensive_health()` to handle UNAVAILABLE checks and justify domain weights.

**Files:**
- `api/services/health_service.py` — update scoring logic

**Implementation notes:**
- Current issue: all checks weighted equally within domain, UNAVAILABLE (score=50) inflates domain score
- Fix:
  1. UNAVAILABLE checks excluded from domain average (not counted at all)
  2. If >50% of domain checks are UNAVAILABLE, domain status = UNAVAILABLE
  3. Domain weights with justification:
     - Data (25%): garbage-in-garbage-out, data quality is foundational
     - Signal (25%): model predictions drive all decisions
     - Risk (20%): risk management protects capital
     - Execution (15%): execution costs eat returns
     - Governance (15%): model management prevents decay
  4. Add `checks_available` counter to response
  5. Overall status: use only available checks for score, but flag if many are unavailable

**Verify:**
```bash
python -c "
from quant_engine.api.services.health_service import HealthService
svc = HealthService()
result = svc.compute_comprehensive_health()
print(f'Overall score: {result[\"overall_score\"]:.1f}')
print(f'Checks available: {result.get(\"checks_available\", \"N/A\")}')
"
```

---

### T9: Test all rewritten health checks

**What:** Comprehensive test file for the new health check system.

**Files:**
- `tests/test_health_checks_rewritten.py` — new test file

**Implementation notes:**
- Test cases:
  1. `test_unavailable_not_score_50` — Exception in check → score=0, status=UNAVAILABLE
  2. `test_health_check_result_has_methodology` — Every check returns methodology string
  3. `test_domain_scoring_excludes_unavailable` — UNAVAILABLE checks not in average
  4. `test_signal_decay_with_declining_ic` — Declining IC → FAIL status
  5. `test_signal_decay_with_stable_ic` — Stable IC → PASS status
  6. `test_survivorship_checks_total_ret` — Score based on actual data, not config
  7. `test_execution_quality_needs_tca` — No TCA → UNAVAILABLE
  8. `test_overall_score_calculation` — Weighted average correct with partial availability

**Verify:**
```bash
python -m pytest tests/test_health_checks_rewritten.py -v
```

---

## Validation

### Acceptance criteria
1. No health check returns score=50 on exception (all return UNAVAILABLE with score=0)
2. Every check includes `methodology` and `thresholds` in its response
3. Overall health score excludes UNAVAILABLE checks from average
4. Signal decay check measures rolling IC trend (not prediction error autocorrelation)
5. Execution quality check uses backtest TCA data (not paper trading simulation)
6. Survivorship check validates data has `total_ret` column (not just WRDS config flag)

### Rollback plan
- Each check is independent — revert individual methods if issues
- HealthCheckResult dataclass is additive — old dict format can coexist
- Domain weights are config-level — easy to adjust
