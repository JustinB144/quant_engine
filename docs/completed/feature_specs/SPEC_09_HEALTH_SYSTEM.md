# Feature Spec: Health System Renaissance Overhaul

> **Status:** Draft
> **Author:** Claude Opus 4
> **Date:** 2026-02-26
> **Estimated effort:** 100 hours across 6 tasks

---

## Why

The health monitoring system in `/mnt/quant_engine/engine/health_service.py` is **already comprehensive** (15 checks, 5 domains, severity-weighted scoring). However, it has four critical gaps:

1. **Information Ratio tracking** is referenced but not implemented. IR is fundamental for alpha signal quality.
2. **Survivorship bias** is checked binary (DB exists?) but not quantified. We need to measure how many signals are lost to delisted securities.
3. **Health score history and trending** are not surfaced to the frontend. Dashboards cannot track whether health is improving or degrading.
4. **No feedback loop to risk governor.** Health scores are computed but don't feed back into sizing/gating decisions. A degraded health score should automatically reduce position sizes.
5. **Health scores lack confidence intervals.** A score based on 5 trades has vastly different uncertainty than a score based on 500 trades.

Furthermore, the domain weighting (25% signal quality, 25% data integrity, 20% risk management, 15% execution, 15% governance) is reasonable, but **execution quality at 15% is too low for a live trading system.** Execution slippage and market impact are live risks; they should be 20–25%.

This spec implements **Information Ratio, quantified survivorship bias, health→risk feedback loop, alert system, and confidence intervals.**

---

## What

Implement a **Health System with Feedback Integration** that:

1. **Tracks Information Ratio** (alpha generation quality):
   - Compute IR = (signal_return - benchmark_return) / tracking_error.
   - Daily rolling IR (20-day window) compared to baseline IR from training.
   - Flag if rolling IR < baseline IR * 0.7 (30% degradation).
   - Threshold: IR < 0.5 is poor, 0.5–1.0 is acceptable, > 1.0 is good.

2. **Quantifies survivorship bias:**
   - Track N_deleted (securities deleted from universe due to bankruptcy/delisting).
   - Compute PnL lost from deleted securities.
   - Report: {N_deleted, pnl_lost_pct: PnL_from_deleted / total_PnL}.
   - Score: if PnL_lost > 5% of total, survivorship bias is significant.

3. **Computes health confidence intervals:**
   - Each health score is based on N samples (trades, days, observations).
   - Report score ± CI (95%) using empirical bootstrap or normal approximation.
   - Highlight scores with low confidence (N < 20).

4. **Surfaces health history and trending:**
   - Store daily health scores (one row per date) in time-series database (e.g., InfluxDB or SQLite).
   - Compute 7-day, 30-day rolling average health.
   - Expose via API: `/health/history?start_date=2026-01-01&end_date=2026-02-26` → time series.
   - Detect trends: is health improving, stable, or degrading?

5. **Implements health→risk feedback:**
   - Feed health score into risk governor.
   - If overall health < 0.6 (6/10), reduce position sizes by 20%.
   - If health < 0.4, reduce by 50% or halt trading.
   - Thresholds configurable via `/mnt/quant_engine/config/health.yaml`.

6. **Implements alert and notification system:**
   - Alert if health drops > 10% day-over-day (sign of sudden degradation).
   - Alert if specific domain score falls below critical threshold (e.g., data integrity < 0.5).
   - Send alerts via logging, email (optional), or webhook.
   - Alert types: CRITICAL, STANDARD, INFORMATIONAL with severity weights (3x, 1x, 0.5x as existing).

---

## Constraints

### Must-haves

- Information Ratio calculated as (signal_return - benchmark_return) / tracking_error, 20-day rolling window.
- Survivorship bias quantified as % PnL lost from deleted securities (not binary check).
- All health scores report 95% CI or sample size N.
- Health scores persisted daily with timestamp in time-series database.
- Health score feeds into risk governor: position size multiplier = f(health_score).
- Alerts triggered for health degradation (> 10% drop) and domain failures (specific score < threshold).
- Execution quality domain weight increased to 20% (from 15%); redistribute other domains.

### Must-nots

- **Do not** invent new IR formula; use standard definition (signal return - benchmark) / tracking error.
- **Do not** implement new database; use existing infrastructure (SQLite or InfluxDB if available).
- **Do not** modify existing health checks in health_service.py unless fixing bugs; only add new checks (IR, quantified survivorship).
- **Do not** gate trading binary on health < 0.6; reduce sizes proportionally instead.
- **Do not** ignore confidence intervals; always report N_samples with score.

### Out of scope

- Backtesting health scores on historical data (separate effort: retrain entire backtest with health calculations).
- ML-based anomaly detection in health scores (classification of "normal" vs. "anomalous" health trajectory). Future: unsupervised learning.
- Multi-agent health consensus (if multiple trading pods, aggregate health). Single-pod focus for this spec.
- Custom alert routing (e.g., PagerDuty integration). Logging and webhook are sufficient.

---

## Current State

### Key files

- **`/mnt/quant_engine/engine/health_service.py`** (80KB+): 15 checks, 5 domains (Data Integrity 25%, Signal Quality 25%, Risk Management 20%, Execution Quality 15%, Model Governance 15%). Severity-weighted scoring (critical 3x, standard 1x, informational 0.5x). Base score removed (starts at 0).
- **Checks implemented:** signal_decay, feature_importance_drift, regime_transition_health, prediction_distribution, survivorship_bias (binary), correlation_regime, execution_quality, tail_risk, cv_gap_trend, data_quality_anomalies, ensemble_disagreement, market_microstructure, retraining_effectiveness, capital_utilization. **Missing:** information_ratio.
- **`/mnt/quant_engine/engine/regime/detector.py`**: Detects regime.
- **`/mnt/quant_engine/engine/trainer.py`**: Retraining logic; logs feature importance.
- **`/mnt/quant_engine/engine/paper_trader.py`**: Records trades, positions, market data.
- **`/mnt/quant_engine/config/health.yaml`** (new): Configuration for health thresholds and alert policy.

### Existing patterns to follow

- Health check signature: `_check_<name>(self) -> Tuple[float, str, str]` (score 0–1, reason, severity).
- Scoring: 5 domains, each domain has multiple checks, domain score is weighted average of check scores.
- Logging: use `logging.info()`, `logging.warning()` for alerts.
- Configuration: YAML-based, environment overrides.

### Configuration

**New config file: `/mnt/quant_engine/config/health.yaml`**

```yaml
domain_weights:
  data_integrity: 0.25
  signal_quality: 0.25
  risk_management: 0.20
  execution_quality: 0.20  # increased from 0.15
  model_governance: 0.10   # reduced from 0.15

check_thresholds:
  information_ratio:
    good: 1.0
    acceptable: 0.5
    poor: 0.0
    min_observations: 20

  survivorship_bias:
    max_pnl_loss_pct: 0.05  # flag if > 5%

  signal_decay:
    ic_threshold: 0.3
    min_observations: 60

  tail_risk:
    skewness_threshold: -0.5  # skewness < -0.5 is risk
    kurtosis_threshold: 4.0   # kurtosis > 4 is risk

severity_weights:
  critical: 3.0
  standard: 1.0
  informational: 0.5

risk_feedback:
  enabled: true
  health_to_size_multiplier:
    health_1_0: 1.0   # health = 100% → 100% size
    health_0_8: 0.95  # health = 80% → 95% size
    health_0_6: 0.80  # health = 60% → 80% size
    health_0_4: 0.50  # health = 40% → 50% size
    health_0_2: 0.20  # health = 20% → 20% size
    health_0_0: 0.05  # health = 0% → 5% size (halt trading)

alerts:
  enabled: true
  health_degradation_threshold: 0.10  # alert if health drops > 10% day-over-day
  domain_critical_threshold: 0.50     # alert if any domain score < 50%
  notification_channels:
    - type: "logging"
      level: "warning"
    - type: "email"
      enabled: false
      recipients: ["ops@example.com"]
    - type: "webhook"
      enabled: false
      url: "https://hooks.slack.com/..."

persistence:
  enabled: true
  backend: "sqlite"  # or "influxdb"
  retention_days: 90
  sqlite_path: "/mnt/quant_engine/data/health_history.db"

reporting:
  health_score_format: "{score:.2f} (N={n_samples}, CI=[{ci_lower:.2f}, {ci_upper:.2f}])"
  include_domain_breakdown: true
  include_check_details: true
```

---

## Tasks

### T1: Information Ratio Tracking

**What:** Implement Information Ratio calculation and trending. Compute rolling IR (20-day window) and compare to baseline IR from training. Add `_check_information_ratio()` to health checks.

**Files:**
- Modify `/mnt/quant_engine/engine/health_service.py`:
  - Add method `_check_information_ratio(self) -> Tuple[float, str, str]`:
    - Compute rolling IR over last 20 days: IR = (signal_return - benchmark_return) / tracking_error.
    - Compare to baseline IR from training phase (stored in trainer logs or config).
    - Score: if IR > baseline * 1.1, score = 1.0; if IR < baseline * 0.7, score = 0.0; linear interpolation in between.
    - Return (score, reason, severity).
  - Helper: `_compute_rolling_ir(signal_returns, benchmark_returns, window=20) -> float`.
  - Add IR check to overall health score computation (map to "signal_quality" domain).
- Compute benchmark returns: use S&P 500 or custom benchmark (configurable).
- Store baseline IR from trainer in config or model metadata.
- Tests: `/mnt/quant_engine/tests/test_health_information_ratio.py`.

**Implementation notes:**
- Signal returns = strategy returns (from paper_trader).
- Benchmark returns = S&P 500 returns (from data source) or 0% (zero benchmark).
- Tracking error = std(signal_return - benchmark_return).
- Baseline IR = IR computed during training window (e.g., last 252 days of backtest).
- Score mapping: IR < 0.0 → score 0, IR = 0.5 → score 0.5, IR > 1.0 → score 1.0.
- Edge case: if window < 20 observations, flag low confidence.

**Verify:**
- Compute IR on synthetic returns. Verify IR computation matches standard formula.
- Test IR trending: decreasing IR over time should lower health score.
- Verify low-N window is flagged with low confidence.

---

### T2: Quantified Survivorship Bias

**What:** Replace binary survivorship check with quantified measurement: track deleted securities, compute PnL lost, and report as % of total PnL.

**Files:**
- Modify `/mnt/quant_engine/engine/health_service.py`:
  - Modify `_check_survivorship_bias(self)` (already exists but binary):
    - Query database for deleted securities (look for securities removed from universe).
    - For each deleted security, look up realized trades and compute cumulative PnL.
    - Compute: pnl_lost_pct = sum_pnl_from_deleted_securities / sum_total_pnl.
    - Score: if pnl_lost_pct < 1%, score = 1.0; if 1–5%, score 0.5; if > 5%, score 0.0.
    - Return (score, reason, severity).
  - Helper: `_identify_deleted_securities() -> List[str]` (query universe changes over time).
  - Helper: `_compute_pnl_from_deleted(securities: List[str]) -> float`.
- Tests: `/mnt/quant_engine/tests/test_health_survivorship_bias.py`.

**Implementation notes:**
- Deleted securities: look for delisting events in market data (price becomes NaN, volume = 0).
- PnL computation: sum realized PnL from all trades in deleted security.
- Scoring: survivorship bias > 5% of PnL is significant (signals regime shift or poor universe selection).
- Edge case: if no trades in deleted security, PnL = 0 (no impact).

**Verify:**
- Synthetic data: create universe with one security that delists mid-backtest. Verify PnL loss is computed.
- Real data: identify delisted securities in S&P 500 (e.g., Lehman 2008). Verify PnL impact is captured.

---

### T3: Health Score Confidence Intervals

**What:** For each health score, compute 95% confidence interval using bootstrap or normal approximation. Report score ± CI.

**Files:**
- Create `/mnt/quant_engine/engine/health_confidence.py` (new):
  - Class `HealthConfidenceCalculator`:
    - `__init__()`: Initialize.
    - `compute_ci_bootstrap(samples: np.ndarray, ci: float = 0.95, n_boot: int = 1000) -> Tuple[float, float]`:
      - Resample samples with replacement 1000 times.
      - Compute mean each time.
      - Return [quantile((1-ci)/2), quantile(1-(1-ci)/2)].
    - `compute_ci_normal(mean: float, std: float, n: int, ci: float = 0.95) -> Tuple[float, float]`:
      - Use t-distribution (not z) for small N.
      - CI = mean ± t_crit(n-1, ci) * std / sqrt(n).
    - `compute_ci(samples_or_params, method: str = "auto") -> Tuple[float, float]`:
      - Auto-select: if N < 30, use t; if N >= 30, use z-approximation.
- Modify `/mnt/quant_engine/engine/health_service.py`:
  - Each `_check_<name>()` method now returns: (score, ci_lower, ci_upper, reason, severity, n_samples).
  - Pass samples to `HealthConfidenceCalculator.compute_ci()`.
  - Aggregate domain scores with CI propagation (for weighted averages, compute CI of combined score).
- Tests: `/mnt/quant_engine/tests/test_health_confidence.py`.

**Implementation notes:**
- Bootstrap is robust but slow (~1000 iterations). Use for small N (< 50).
- Normal approximation (t-distribution) is fast and accurate for N >= 20.
- For binary checks (pass/fail), use binomial CI.
- Propagate CI through weighted average: if domain score = w1*s1 + w2*s2, CI = sqrt((w1*ci1)^2 + (w2*ci2)^2).
- Report in config as: "{score:.2f} ± {ci:.2f} (N={n})".

**Verify:**
- Synthetic data: 1000 Bernoulli samples (prob 0.6). Compute CI; verify true prob 0.6 is inside CI.
- Real data: compute CI on rolling 20-day information ratio; verify CI width decreases with N.

---

### T4: Health History and Trending

**What:** Persist daily health scores to time-series database. Compute rolling averages and trends. Expose via API.

**Files:**
- Create `/mnt/quant_engine/engine/health_database.py` (new):
  - Class `HealthHistoryDB`:
    - `__init__(backend: str = "sqlite", path: str = None)`: Connect to SQLite or InfluxDB.
    - `store_daily_health(date: datetime, health_score: float, domain_scores: Dict, check_details: Dict)`:
      - Insert row: {date, overall_health, data_integrity, signal_quality, risk_management, execution_quality, model_governance, ...details}.
      - Implemented for both SQLite (relational) and InfluxDB (time-series).
    - `fetch_health_history(start_date, end_date) -> pd.DataFrame`: Return time series.
    - `compute_rolling_average(window: int = 7) -> pd.Series`: Compute 7-day MA.
    - `detect_trend(health_series: pd.Series, window: int = 30) -> Tuple[bool, str]`: Linear regression. Return (improving, degrading, stable).
  - SQLite schema:
    ```sql
    CREATE TABLE health_history (
      date TEXT PRIMARY KEY,
      overall_health REAL,
      data_integrity REAL,
      signal_quality REAL,
      risk_management REAL,
      execution_quality REAL,
      model_governance REAL,
      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    ```
- Modify `/mnt/quant_engine/engine/health_service.py`:
  - On daily health computation, call `health_db.store_daily_health()`.
- Create `/mnt/quant_engine/engine/health_api.py` (new):
  - Flask/FastAPI endpoint: `GET /health/history?start_date=2026-01-01&end_date=2026-02-26`:
    - Return JSON: {dates: [...], scores: [...], rolling_7d: [...], trend: "improving"}.
- Tests: `/mnt/quant_engine/tests/test_health_database.py`.

**Implementation notes:**
- SQLite is simple, no external dependency. InfluxDB is scalable but requires setup.
- Store full domain breakdown and check details (for debugging).
- Rolling average: 7-day MA is standard (removes noise).
- Trend detection: linear regression on 30-day window. Slope > 0.01 = improving, < -0.01 = degrading.
- Retention: keep 90 days of history (configurable).

**Verify:**
- Store daily health scores for 30 days. Fetch via API; verify time series is correct.
- Compute 7-day MA; verify it's less noisy than daily.
- Simulate improving trend (daily scores 0.5 → 0.9). Detect trend = improving.

---

### T5: Health to Risk Feedback Loop

**What:** Feed health score into risk governor. Implement `health_to_size_multiplier()` that scales position sizes based on health.

**Files:**
- Create `/mnt/quant_engine/engine/health_risk_feedback.py` (new):
  - Class `HealthRiskGate`:
    - `__init__(config: Dict)`: Load health_to_size_multiplier from health.yaml.
    - `compute_size_multiplier(health_score: float) -> float`:
      - Interpolate from health_to_size_multiplier table.
      - Example: health=0.6 → multiplier=0.8 (reduce sizes by 20%).
      - Return multiplier ∈ [0, 1].
    - `apply_health_gate(weights: np.ndarray, health_score: float) -> np.ndarray`:
      - Compute multiplier = `compute_size_multiplier(health_score)`.
      - Return weights * multiplier.
    - `should_halt_trading(health_score: float) -> bool`:
      - Return True if health < min_health (e.g., 0.05 → halt).
- Modify `/mnt/quant_engine/engine/paper_trader.py` or position sizer:
  - After computing recommended weights, apply `health_risk_gate.apply_health_gate()`.
  - If `should_halt_trading()`, set all weights to 0 and log alert.
- Tests: `/mnt/quant_engine/tests/test_health_risk_feedback.py`.

**Implementation notes:**
- Size multiplier is linear interpolation between table points in health.yaml.
- Example config:
  - health 1.0 → multiplier 1.0 (full size).
  - health 0.6 → multiplier 0.8 (20% reduction).
  - health 0.0 → multiplier 0.05 (near-halt, emergency orders only).
- If health drops suddenly, sizes gradually reduce (exponential smoothing over 1–2 days).
- Logging: log multiplier applied so trader can see impact.

**Verify:**
- Set health = 0.8, verify size multiplier = 0.95.
- Set health = 0.2, verify trading halts (weight multiplier ≈ 0).
- Simulate health degradation 1.0 → 0.4 over 5 days; verify sizes reduce smoothly (not step change).

---

### T6: Alert and Notification System

**What:** Implement alert logic for health degradation and domain failures. Send alerts via logging, email (optional), webhook.

**Files:**
- Create `/mnt/quant_engine/engine/health_alerts.py` (new):
  - Class `HealthAlertManager`:
    - `__init__(config: Dict)`: Load alert thresholds and channels from health.yaml.
    - `check_health_degradation(health_today: float, health_yesterday: float) -> Optional[Alert]`:
      - If health drops > 10%, return Alert(type=CRITICAL, message="Health degraded 15% day-over-day").
    - `check_domain_failures(domain_scores: Dict) -> List[Alert]`:
      - If any domain score < 50%, return Alert(type=CRITICAL, message=f"{domain} health < 50%").
    - `send_alert(alert: Alert, channels: List[str])`:
      - Log via logging module.
      - If email enabled, send to recipients.
      - If webhook enabled, POST to webhook URL.
  - Class `Alert`: Dataclass with {type, message, timestamp, domain}.
- Modify `/mnt/quant_engine/engine/health_service.py`:
  - On daily health computation, call `alert_manager.check_health_degradation()` and `check_domain_failures()`.
  - Send alerts if triggered.
- Tests: `/mnt/quant_engine/tests/test_health_alerts.py`.

**Implementation notes:**
- Alert types: CRITICAL (health < 0.4 or domain < 0.5), STANDARD (health degradation > 10%), INFORMATIONAL (other events).
- Logging via `logging.critical()`, `logging.warning()`, `logging.info()`.
- Email and webhook are optional; disable if not configured.
- Deduplicate: if same alert triggered yesterday, don't send again (rate limiting).
- Webhook payload: `{timestamp, alert_type, message, health_score, domain_scores}`.

**Verify:**
- Simulate health degradation 0.8 → 0.65 (18% drop). Verify CRITICAL alert sent.
- Simulate domain failure: data_integrity = 0.3. Verify STANDARD alert sent.
- Verify logging captures all alerts.

---

## Validation

### Acceptance criteria

1. **Information Ratio implemented:** `_check_information_ratio()` computes rolling IR, compares to baseline, scores from 0–1.
2. **Survivorship bias quantified:** `_check_survivorship_bias()` returns % PnL lost (not binary). Scores based on pnl_lost_pct threshold.
3. **Confidence intervals:** All health scores report 95% CI. Bootstrap or normal approximation used based on N. Low-N scores flagged.
4. **Health history persisted:** Daily health scores stored in SQLite. `HealthHistoryDB.fetch_history()` returns time series. API endpoint `/health/history` works.
5. **Trend detection:** `detect_trend()` correctly identifies improving (slope > 0.01), degrading (slope < -0.01), stable (|slope| ≤ 0.01).
6. **Health to risk feedback:** `HealthRiskGate.apply_health_gate()` scales positions. health 0.6 → multiplier 0.8. Trading halted if health < min_health.
7. **Alerts triggered:** Health degradation > 10% triggers CRITICAL alert. Domain failure (< 50%) triggers STANDARD alert. Logging captures all alerts.
8. **Domain weights updated:** Execution quality 20%, Model Governance 10% (other domains adjusted). Sum = 100%.
9. **Tests pass:** All unit tests pass. Coverage > 85%.
10. **Documentation:** `/mnt/quant_engine/docs/health_guide.md` documents new checks, CI reporting, risk feedback, alerts.

### Verification steps

1. Compute IR on 20-day window of returns. Verify IR formula is correct (compared to external tool).
2. Identify delisted securities in backtest window. Compute PnL lost; verify reported as % of total.
3. Run health computation on synthetic data with known scores. Verify CI bounds contain true score.
4. Store health scores for 30 days; fetch via API. Verify time series is correct.
5. Compute 7-day MA; verify it's smoother than daily.
6. Simulate health improvement trend. Run `detect_trend()`; verify returns "improving".
7. Set health = 0.6, compute size multiplier. Verify multiplier = 0.8.
8. Simulate health degradation 0.9 → 0.7 (20% drop). Verify CRITICAL alert logged.
9. Simulate domain failure: execution_quality = 0.3. Verify alert triggered.
10. Run pytest on test suite. Verify coverage > 85%.

### Rollback plan

- **If IR calculation fails:** Fall back to correlation of returns and predictions (simpler metric, no benchmark needed).
- **If survivorship bias computation is slow:** Cache deleted securities list (update daily).
- **If CI computation is slow:** Use normal approximation only (skip bootstrap).
- **If database insertion fails:** Fall back to CSV file logging (slower but reliable).
- **If health→risk feedback causes excessive position reduction:** Disable feedback and rely on manual trading decisions.
- **If alert system spams alerts:** Increase degradation threshold (e.g., 20% instead of 10%) or implement rate limiting.
- **If tests fail:** Revert to previous version of health_service.py.

---

## Notes

- **Information Ratio is signal-dependent:** If strategy has no alpha (IR = 0), health score for this check is 0. This is correct; IR directly measures alpha generation.
- **Survivorship bias is often overlooked:** Many backtests ignore delisted securities. Quantifying impact is essential for realistic performance.
- **Confidence intervals should guide decision-making:** A health score of 0.6 with CI=[0.4, 0.8] (wide) is more uncertain than 0.6 ± [0.58, 0.62] (tight). Use N and CI to weight decisions.
- **Health history provides valuable context:** A single-day health dip (0.8 → 0.6) is less concerning than sustained degradation (0.8 → 0.7 → 0.6 → 0.5). Trends matter more than points.
- **Health→risk feedback is conservative:** At health = 0.6, reduce by only 20%, not 50%. Prevents overreaction to temporary noise.
- **Execution quality importance:** Live trading is exposed to real slippage, market impact, rejection rates. Execution quality at 20% (vs. 15%) reflects this reality.
- **Future extension: ML-based anomaly detection:** Use autoencoder or isolation forest on health trajectory to detect regime changes earlier than simple trending.
