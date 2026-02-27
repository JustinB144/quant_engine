"""System health assessment for API consumption.

Expanded health service with 17 runtime monitoring checks across 5 domains:
    - Data Integrity (25%)
    - Signal Quality (25%)  — includes IR (Spec 09) + IC tracking (SPEC-H01)
    - Risk Management (20%)
    - Execution Quality (20%)  — increased from 15% (Spec 09)
    - Model Governance (10%)   — reduced from 15% (Spec 09)

Each check returns a ``HealthCheckResult`` dataclass with score, status,
methodology, and raw metrics.  UNAVAILABLE checks (missing data) score 0
and are excluded from domain averages so they don't inflate results.

Spec 09 additions:
    - Information Ratio check (rolling 20-day IR vs baseline)
    - Quantified survivorship bias (PnL loss from deleted securities)
    - Confidence intervals on all health scores
    - Enhanced health history with rolling averages and trend detection
    - Health alert integration (degradation + domain failure detection)
    - Health-to-risk feedback loop (position size scaling)

SPEC-H01 additions:
    - IC tracking across autopilot cycles (rolling mean + trend)
    - IC history persistence in SQLite (ic_tracking.db)
    - WARNING if rolling IC mean < 0.01, CRITICAL if IC < 0

SPEC-H02 additions:
    - Real-time ensemble disagreement monitoring (std of member predictions)
    - Disagreement history persistence in SQLite (disagreement_tracking.db)
    - Rolling mean disagreement with trend detection
    - WARNING if mean disagreement > 0.015, CRITICAL if > 0.03
    - Static holdout_corr fallback when no tracking data exists

SPEC-H03 additions:
    - Execution quality monitoring via cost surprise tracking
    - Paper trade fill quality persistence in SQLite (exec_quality.db)
    - Rolling cost surprise distribution (actual - predicted cost in bps)
    - WARNING if mean cost surprise > 2 bps (model underestimates costs)
    - CRITICAL if mean cost surprise > 5 bps
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ── HealthCheckResult dataclass ─────────────────────────────────────────

_SEVERITY_WEIGHTS = {"critical": 3.0, "standard": 1.0, "informational": 0.5}


@dataclass
class HealthCheckResult:
    """Structured result from a single health check."""

    name: str
    domain: str
    score: float  # 0–100
    status: str  # PASS, WARN, FAIL, UNAVAILABLE
    explanation: str = ""
    methodology: str = ""
    data_available: bool = True
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    severity: str = "standard"  # "critical", "standard", "informational"
    raw_value: Optional[float] = None  # The actual measured value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "score": self.score,
            "status": self.status,
            "explanation": self.explanation,
            "methodology": self.methodology,
            "data_available": self.data_available,
            "raw_metrics": self.raw_metrics,
            "thresholds": self.thresholds,
            "severity": self.severity,
            "raw_value": self.raw_value,
        }


def _unavailable(name: str, domain: str, reason: str) -> HealthCheckResult:
    """Return a standard UNAVAILABLE result (score=0, excluded from averages)."""
    return HealthCheckResult(
        name=name,
        domain=domain,
        score=0.0,
        status="UNAVAILABLE",
        explanation=reason,
        methodology="N/A — data not available",
        data_available=False,
    )


class HealthService:
    """Computes system health from model age, cache freshness, and runtime metrics."""

    def get_quick_status(self) -> Dict[str, Any]:
        """Lightweight health check for ``GET /api/health``."""
        from quant_engine.config import DATA_CACHE_DIR, MODEL_DIR

        status = "healthy"
        checks: Dict[str, str] = {}

        # Cache freshness
        cache_dir = Path(DATA_CACHE_DIR)
        if cache_dir.exists():
            parquets = list(cache_dir.glob("*.parquet"))
            if parquets:
                ages = [
                    (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
                    for f in parquets[:10]
                ]
                max_age = max(ages)
                checks["cache_max_age_days"] = str(max_age)
                if max_age > 21:
                    status = "degraded"
            else:
                checks["cache"] = "no_parquet_files"
                status = "degraded"
        else:
            checks["cache"] = "missing"
            status = "unhealthy"

        # Model availability
        model_dir = Path(MODEL_DIR)
        registry_path = model_dir / "registry.json"
        if registry_path.exists():
            checks["model_registry"] = "present"
        else:
            checks["model_registry"] = "missing"
            if status == "healthy":
                status = "degraded"

        # WRDS data source availability
        try:
            from quant_engine.config import WRDS_ENABLED
            if WRDS_ENABLED:
                from quant_engine.data.wrds_provider import WRDSProvider
                provider = WRDSProvider()
                if provider.available():
                    checks["wrds"] = "connected"
                else:
                    checks["wrds"] = "unavailable"
                    if status == "healthy":
                        status = "degraded"
            else:
                checks["wrds"] = "disabled"
        except Exception as e:
            checks["wrds"] = f"error: {e}"
            if status == "healthy":
                status = "degraded"

        return {"status": status, "checks": checks, "timestamp": datetime.now(timezone.utc).isoformat()}

    def get_detailed_health(self) -> Dict[str, Any]:
        """Full system health assessment."""
        from .data_helpers import collect_health_data

        payload = collect_health_data()
        result: Dict[str, Any] = {
            "overall_score": payload.overall_score,
            "overall_status": payload.overall_status,
            "generated_at": payload.generated_at.isoformat(),
            "data_integrity_score": payload.data_integrity_score,
            "promotion_score": payload.promotion_score,
            "wf_score": payload.wf_score,
            "execution_score": payload.execution_score,
            "complexity_score": payload.complexity_score,
        }
        for section in [
            "survivorship_checks", "data_quality_checks", "promotion_checks",
            "wf_checks", "execution_checks", "complexity_checks", "strengths",
        ]:
            checks = getattr(payload, section, [])
            result[section] = [
                {"name": c.name, "status": c.status, "detail": c.detail,
                 "value": c.value, "recommendation": c.recommendation}
                for c in checks
            ]
        result["promotion_funnel"] = payload.promotion_funnel
        result["feature_inventory"] = payload.feature_inventory
        result["knob_inventory"] = payload.knob_inventory

        # Attach comprehensive runtime health checks
        runtime_health = self.compute_comprehensive_health()
        result["runtime_health"] = runtime_health

        # Save snapshot for history tracking
        self.save_health_snapshot(runtime_health)

        # Attach history for trend visualization
        result["health_history"] = self.get_health_history()

        return result

    # ── Domain Scoring ──────────────────────────────────────────────────

    @staticmethod
    def _domain_score(checks: List[HealthCheckResult]) -> Optional[float]:
        """Compute severity-weighted mean score, excluding UNAVAILABLE checks.

        Severity weights: critical=3.0, standard=1.0, informational=0.5.
        Returns None if all checks are UNAVAILABLE.
        """
        available = [c for c in checks if c.status != "UNAVAILABLE"]
        if not available:
            return None
        weighted_sum = sum(
            c.score * _SEVERITY_WEIGHTS.get(c.severity, 1.0) for c in available
        )
        weight_total = sum(
            _SEVERITY_WEIGHTS.get(c.severity, 1.0) for c in available
        )
        return float(weighted_sum / weight_total) if weight_total > 0 else 0.0

    @staticmethod
    def _domain_status(checks: List[HealthCheckResult]) -> str:
        """Determine domain-level status."""
        total = len(checks)
        unavailable = sum(1 for c in checks if c.status == "UNAVAILABLE")
        if total > 0 and unavailable / total > 0.5:
            return "UNAVAILABLE"
        available = [c for c in checks if c.status != "UNAVAILABLE"]
        if not available:
            return "UNAVAILABLE"
        avg = float(np.mean([c.score for c in available]))
        if avg >= 75:
            return "PASS"
        if avg >= 50:
            return "WARN"
        return "FAIL"

    # ── Comprehensive Health ────────────────────────────────────────────

    def compute_comprehensive_health(self) -> Dict[str, Any]:
        """Compute comprehensive health score across 5 weighted domains.

        Domain weights (Spec 09 updated):
            - Data Integrity (25%): survivorship + anomalies + microstructure
            - Signal Quality (25%): IC decay + prediction distribution + ensemble + IR
            - Risk Management (20%): tail risk + correlations + capital utilization
            - Execution Quality (20%): execution quality + signal profitability
            - Model Governance (10%): feature drift + CV gap + regime + retraining
        """
        from .health_confidence import HealthConfidenceCalculator, ConfidenceResult

        ci_calc = HealthConfidenceCalculator()
        domains: Dict[str, Dict[str, Any]] = {}

        checks_total = 0
        checks_available = 0

        # Severity classification:
        #   critical  (3x weight) — data integrity and signal quality issues
        #   standard  (1x weight) — risk and execution issues
        #   informational (0.5x weight) — governance/monitoring issues

        # ── Data Integrity (25%) ──
        data_checks = [
            self._check_survivorship_bias(),
            self._check_data_quality_anomalies(),
            self._check_market_microstructure(),
            self._check_wrds_status(),
        ]
        for c in data_checks:
            c.severity = "critical"
        data_score = self._domain_score(data_checks)
        data_status = self._domain_status(data_checks)
        domains["data_integrity"] = {
            "weight": 0.25, "score": data_score, "status": data_status,
            "checks": [c.to_dict() for c in data_checks],
        }

        # ── Signal Quality (25%) — includes IR (Spec 09) + IC tracking (SPEC-H01) ──
        signal_checks = [
            self._check_signal_decay(),
            self._check_prediction_distribution(),
            self._check_ensemble_disagreement(),
            self._check_information_ratio(),
            self._check_ic_tracking(),
        ]
        for c in signal_checks:
            c.severity = "critical"
        signal_score = self._domain_score(signal_checks)
        signal_status = self._domain_status(signal_checks)
        domains["signal_quality"] = {
            "weight": 0.25, "score": signal_score, "status": signal_status,
            "checks": [c.to_dict() for c in signal_checks],
        }

        # ── Risk Management (20%) ──
        risk_checks = [
            self._check_tail_risk(),
            self._check_correlation_regime(),
            self._check_capital_utilization(),
        ]
        for c in risk_checks:
            c.severity = "standard"
        risk_score = self._domain_score(risk_checks)
        risk_status = self._domain_status(risk_checks)
        domains["risk_management"] = {
            "weight": 0.20, "score": risk_score, "status": risk_status,
            "checks": [c.to_dict() for c in risk_checks],
        }

        # ── Execution Quality (20% — increased from 15%, Spec 09) ──
        exec_checks = [
            self._check_execution_quality(),
            self._check_signal_profitability(),
            self._check_cost_surprise(),
        ]
        for c in exec_checks:
            c.severity = "standard"
        exec_score = self._domain_score(exec_checks)
        exec_status = self._domain_status(exec_checks)
        domains["execution_quality"] = {
            "weight": 0.20, "score": exec_score, "status": exec_status,
            "checks": [c.to_dict() for c in exec_checks],
        }

        # ── Model Governance (10% — reduced from 15%, Spec 09) ──
        gov_checks = [
            self._check_feature_importance_drift(),
            self._check_cv_gap_trend(),
            self._check_regime_transition_health(),
            self._check_retraining_effectiveness(),
        ]
        for c in gov_checks:
            c.severity = "informational"
        gov_score = self._domain_score(gov_checks)
        gov_status = self._domain_status(gov_checks)
        domains["model_governance"] = {
            "weight": 0.10, "score": gov_score, "status": gov_status,
            "checks": [c.to_dict() for c in gov_checks],
        }

        # Count available/total
        all_checks = data_checks + signal_checks + risk_checks + exec_checks + gov_checks
        checks_total = len(all_checks)
        checks_available = sum(1 for c in all_checks if c.status != "UNAVAILABLE")

        # Weighted overall — only domains with available data contribute
        available_domains = {
            k: v for k, v in domains.items() if v["score"] is not None
        }
        if available_domains:
            weight_sum = sum(v["weight"] for v in available_domains.values())
            overall = sum(
                v["weight"] * v["score"] for v in available_domains.values()
            ) / weight_sum if weight_sum > 0 else None
        else:
            overall = None

        overall_status = "UNAVAILABLE"
        if overall is not None:
            overall_status = "PASS" if overall >= 75 else "WARN" if overall >= 50 else "FAIL"

        # ── Confidence intervals (Spec 09) ──
        domain_ci: Dict[str, Dict[str, Any]] = {}
        for dname, dinfo in domains.items():
            avail = [
                c for c in dinfo["checks"]
                if c.get("status") != "UNAVAILABLE"
            ]
            scores = [c.get("score", 0.0) for c in avail]
            if scores:
                ci_result = ci_calc.compute_ci(samples=np.array(scores))
                domain_ci[dname] = ci_result.to_dict()
            else:
                domain_ci[dname] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                                    "n_samples": 0, "method": "insufficient",
                                    "low_confidence": True, "ci_width": 0.0}

        # Overall CI: propagate through weighted domains
        overall_ci = None
        if overall is not None and available_domains:
            d_scores = []
            d_widths = []
            d_weights = []
            for dname, dinfo in available_domains.items():
                d_scores.append(dinfo["score"])
                ci = domain_ci.get(dname, {})
                d_widths.append(ci.get("ci_width", 0.0))
                d_weights.append(dinfo["weight"])
            ci_lower, ci_upper = HealthConfidenceCalculator.propagate_weighted_ci(
                d_scores, d_widths, d_weights,
            )
            overall_ci = {
                "ci_lower": round(ci_lower, 1),
                "ci_upper": round(ci_upper, 1),
                "ci_width": round(ci_upper - ci_lower, 1),
            }

        # ── Alerts (Spec 09) ──
        alert_events = self._run_alerts(overall, domains)

        return {
            "overall_score": round(overall, 1) if overall is not None else None,
            "overall_status": overall_status,
            "overall_ci": overall_ci,
            "overall_methodology": (
                "Weighted average of 5 domains: Data Integrity (25%), Signal Quality (25%), "
                "Risk Management (20%), Execution Quality (20%), Model Governance (10%). "
                "Only checks with available data are scored. Within each domain, checks are "
                "weighted by severity: critical (3x), standard (1x), informational (0.5x). "
                "Confidence intervals computed via bootstrap (N<30) or normal approximation."
            ),
            "checks_available": checks_available,
            "checks_total": checks_total,
            "alerts": alert_events,
            "domains": {
                k: {
                    "score": round(v["score"], 1) if v["score"] is not None else None,
                    "weight": v["weight"],
                    "status": v["status"],
                    "ci": domain_ci.get(k),
                    "checks_available": sum(
                        1 for c in v["checks"] if c.get("status") != "UNAVAILABLE"
                    ),
                    "checks_total": len(v["checks"]),
                    "checks": v["checks"],
                }
                for k, v in domains.items()
            },
        }

    # ── Helper: load backtest trades CSV ────────────────────────────────

    @staticmethod
    def _load_trades_csv() -> Optional[pd.DataFrame]:
        """Load backtest_10d_trades.csv, return None if unavailable."""
        try:
            from quant_engine.config import RESULTS_DIR
            path = Path(RESULTS_DIR) / "backtest_10d_trades.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path)
            if df.empty:
                return None
            return df
        except Exception:
            return None

    # ── Signal Quality Checks ───────────────────────────────────────────

    def _check_signal_decay(self) -> HealthCheckResult:
        """Check rolling IC trend — is signal strength declining over time?

        Computes Spearman rank correlation between predicted_return and
        actual_return in rolling windows, then checks if IC_recent / IC_initial
        ratio indicates decay.
        """
        domain = "signal_quality"
        name = "signal_decay"
        methodology = (
            "Rolling IC (Spearman) between predicted and actual returns "
            "in windows of 200 trades.  Ratio of recent IC to initial IC "
            "detects signal decay."
        )
        thresholds = {"ratio_pass": 0.8, "ratio_warn": 0.5}
        try:
            df = self._load_trades_csv()
            if df is None:
                return _unavailable(name, domain, "No backtest trades CSV")
            if "predicted_return" not in df.columns or "actual_return" not in df.columns:
                return _unavailable(name, domain, "Trades CSV missing predicted/actual columns")
            df = df.dropna(subset=["predicted_return", "actual_return"])
            if len(df) < 100:
                return _unavailable(name, domain, f"Only {len(df)} trades with predictions — need 100+")

            # Compute rolling IC in windows
            window = min(200, len(df) // 3)
            if window < 30:
                window = 30
            ics = []
            for start in range(0, len(df) - window + 1, window // 2):
                chunk = df.iloc[start:start + window]
                if len(chunk) >= 20:
                    corr, _ = sp_stats.spearmanr(chunk["predicted_return"], chunk["actual_return"])
                    if not np.isnan(corr):
                        ics.append(corr)

            if len(ics) < 2:
                return _unavailable(name, domain, "Not enough windows for IC trend")

            ic_initial = float(np.mean(ics[:max(1, len(ics) // 3)]))
            ic_recent = float(np.mean(ics[-max(1, len(ics) // 3):]))
            overall_ic = float(np.mean(ics))

            # Avoid division by zero — use absolute comparison when initial IC near zero
            if abs(ic_initial) < 1e-6:
                ratio = 1.0 if abs(ic_recent) < 1e-6 else 0.0
            else:
                ratio = ic_recent / ic_initial

            raw = {"ic_initial": round(ic_initial, 4), "ic_recent": round(ic_recent, 4),
                   "ratio": round(ratio, 4), "overall_ic": round(overall_ic, 4),
                   "n_windows": len(ics)}

            if ratio > 0.8:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"IC stable (ratio {ratio:.2f}, recent={ic_recent:.4f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if ratio > 0.5:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"IC declining (ratio {ratio:.2f}, recent={ic_recent:.4f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"IC decayed significantly (ratio {ratio:.2f}, recent={ic_recent:.4f})",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_prediction_distribution(self) -> HealthCheckResult:
        """Detect variance collapse in predictions using std AND IQR."""
        domain = "signal_quality"
        name = "prediction_distribution"
        methodology = (
            "Check std AND IQR of predicted returns from backtest trades. "
            "Both must exceed minimum thresholds to confirm predictions are "
            "not collapsing to a single value."
        )
        thresholds = {"std_pass": 0.005, "iqr_pass": 0.003}
        try:
            df = self._load_trades_csv()
            if df is None:
                return _unavailable(name, domain, "No backtest trades CSV")
            if "predicted_return" not in df.columns:
                return _unavailable(name, domain, "No predicted_return column")

            preds = pd.to_numeric(df["predicted_return"], errors="coerce").dropna()
            if len(preds) < 10:
                return _unavailable(name, domain, "Too few predictions")

            std_val = float(preds.std())
            q75, q25 = float(preds.quantile(0.75)), float(preds.quantile(0.25))
            iqr_val = q75 - q25

            raw = {"std": round(std_val, 6), "iqr": round(iqr_val, 6),
                   "n_predictions": len(preds), "mean": round(float(preds.mean()), 6)}

            if std_val > 0.005 and iqr_val > 0.003:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"Healthy spread (std={std_val:.6f}, IQR={iqr_val:.6f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if std_val < 1e-6:
                return HealthCheckResult(
                    name=name, domain=domain, score=10.0, status="FAIL",
                    explanation=f"Variance collapse (std={std_val:.6f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=45.0, status="WARN",
                explanation=f"Low prediction variance (std={std_val:.6f}, IQR={iqr_val:.6f})",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_ensemble_disagreement(self) -> HealthCheckResult:
        """Track ensemble member prediction disagreement over time (SPEC-H02).

        Primary: reads from the disagreement_history database populated by
        the autopilot engine after each prediction cycle.  Computes rolling
        mean and trend of ensemble disagreement (std of member predictions).

        Fallback: if no tracking history exists, falls back to static model
        metadata analysis (holdout_corr spread across regime models).

        Score mapping:
            mean_disagreement <= low_thresh  -> 90 (PASS, ensemble aligned)
            mean_disagreement <= warn_thresh -> 70 (PASS, moderate)
            mean_disagreement <= crit_thresh -> 45 (WARN, high disagreement)
            mean_disagreement >  crit_thresh -> 20 (FAIL, critical)
        """
        domain = "signal_quality"
        name = "ensemble_disagreement"
        methodology = (
            "Track per-prediction ensemble member disagreement (std of global "
            "and regime model predictions) over time.  Low disagreement means "
            "ensemble members agree on direction/magnitude.  High disagreement "
            "indicates signal unreliability and increased uncertainty.  Falls "
            "back to static holdout_corr analysis when no tracking data exists."
        )

        try:
            from quant_engine.config import (
                ENSEMBLE_DISAGREEMENT_LOOKBACK,
                ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD,
                ENSEMBLE_DISAGREEMENT_CRITICAL_THRESHOLD,
            )
        except ImportError:
            ENSEMBLE_DISAGREEMENT_LOOKBACK = 20
            ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD = 0.015
            ENSEMBLE_DISAGREEMENT_CRITICAL_THRESHOLD = 0.03

        low_threshold = ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD * 0.5
        thresholds = {
            "low": round(low_threshold, 4),
            "warn": ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD,
            "critical": ENSEMBLE_DISAGREEMENT_CRITICAL_THRESHOLD,
        }

        try:
            # Primary: read from disagreement history database
            history = self.get_disagreement_history(
                limit=ENSEMBLE_DISAGREEMENT_LOOKBACK,
            )
            disagreement_values = [
                h["mean_disagreement"] for h in history
                if h["mean_disagreement"] is not None
            ]

            if disagreement_values:
                d_arr = np.array(disagreement_values, dtype=float)
                rolling_mean = float(np.mean(d_arr))
                d_std = float(np.std(d_arr)) if len(d_arr) > 1 else 0.0

                # Trend detection via linear regression
                trend_label = "unknown"
                trend_slope = 0.0
                if len(d_arr) >= 3:
                    x = np.arange(len(d_arr), dtype=float)
                    coeffs = np.polyfit(x, d_arr, 1)
                    trend_slope = float(coeffs[0])
                    if trend_slope > 0.0005:
                        trend_label = "increasing"
                    elif trend_slope < -0.0005:
                        trend_label = "decreasing"
                    else:
                        trend_label = "stable"

                latest_pct_high = (
                    history[-1].get("pct_high_disagreement", 0.0)
                    if history else 0.0
                )

                raw = {
                    "rolling_mean_disagreement": round(rolling_mean, 6),
                    "disagreement_std": round(d_std, 6),
                    "n_snapshots": len(disagreement_values),
                    "trend": trend_label,
                    "trend_slope": round(trend_slope, 6),
                    "latest_disagreement": round(float(d_arr[-1]), 6),
                    "latest_pct_high": round(float(latest_pct_high), 4),
                    "n_members": history[-1].get("n_members", 0) if history else 0,
                    "member_names": history[-1].get("member_names", []) if history else [],
                    "disagreement_values": [round(float(v), 6) for v in d_arr],
                }

                # Scoring: lower disagreement is better
                if rolling_mean <= low_threshold:
                    return HealthCheckResult(
                        name=name, domain=domain, score=90.0, status="PASS",
                        explanation=(
                            f"Ensemble well-aligned: mean disagreement="
                            f"{rolling_mean:.4f} over {len(disagreement_values)} "
                            f"cycles (trend: {trend_label})"
                        ),
                        methodology=methodology, raw_metrics=raw,
                        thresholds=thresholds, raw_value=rolling_mean,
                    )
                if rolling_mean <= ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD:
                    return HealthCheckResult(
                        name=name, domain=domain, score=70.0, status="PASS",
                        explanation=(
                            f"Moderate ensemble disagreement: mean="
                            f"{rolling_mean:.4f} over {len(disagreement_values)} "
                            f"cycles (trend: {trend_label})"
                        ),
                        methodology=methodology, raw_metrics=raw,
                        thresholds=thresholds, raw_value=rolling_mean,
                    )
                if rolling_mean <= ENSEMBLE_DISAGREEMENT_CRITICAL_THRESHOLD:
                    return HealthCheckResult(
                        name=name, domain=domain, score=45.0, status="WARN",
                        explanation=(
                            f"High ensemble disagreement: mean="
                            f"{rolling_mean:.4f} > {ENSEMBLE_DISAGREEMENT_WARN_THRESHOLD} "
                            f"over {len(disagreement_values)} cycles "
                            f"(trend: {trend_label}) — signal may be unreliable"
                        ),
                        methodology=methodology, raw_metrics=raw,
                        thresholds=thresholds, raw_value=rolling_mean,
                    )
                return HealthCheckResult(
                    name=name, domain=domain, score=20.0, status="FAIL",
                    explanation=(
                        f"CRITICAL ensemble disagreement: mean="
                        f"{rolling_mean:.4f} > {ENSEMBLE_DISAGREEMENT_CRITICAL_THRESHOLD} "
                        f"over {len(disagreement_values)} cycles "
                        f"(trend: {trend_label}) — ensemble unreliable"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, severity="critical",
                    raw_value=rolling_mean,
                )

            # Fallback: static model metadata analysis
            return self._check_ensemble_disagreement_static_fallback(
                name, domain, methodology, thresholds,
            )

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_ensemble_disagreement_static_fallback(
        self,
        name: str,
        domain: str,
        methodology: str,
        thresholds: Dict[str, Any],
    ) -> HealthCheckResult:
        """Fallback: analyze static holdout_corr from model metadata.

        Used when no prediction-level disagreement tracking data exists.
        """
        from quant_engine.config import MODEL_DIR
        meta_files = sorted(Path(MODEL_DIR).rglob("*_meta.json"), reverse=True)
        if not meta_files:
            return _unavailable(name, domain, "No model metadata or tracking history found")

        with open(meta_files[0], "r") as f:
            meta = json.load(f)

        regime_models = meta.get("regime_models", {})
        if not regime_models:
            return HealthCheckResult(
                name=name, domain=domain, score=60.0, status="WARN",
                explanation="Only global model — no regime ensemble to compare",
                methodology=methodology + " (static fallback — no tracking data)",
                data_available=True,
            )

        holdout_corrs = [
            float(v.get("holdout_corr", 0)) for v in regime_models.values()
            if "holdout_corr" in v
        ]
        if not holdout_corrs:
            return _unavailable(
                name, domain,
                "No holdout correlations or tracking history",
            )

        spread = float(max(holdout_corrs) - min(holdout_corrs))
        avg = float(np.mean(holdout_corrs))
        raw = {
            "avg_holdout_corr": round(avg, 4),
            "spread": round(spread, 4),
            "n_regime_models": len(holdout_corrs),
            "per_regime": {
                k: round(float(v.get("holdout_corr", 0)), 4)
                for k, v in regime_models.items()
            },
            "source": "static_fallback",
        }

        if avg > 0.02 and spread < 0.20:
            return HealthCheckResult(
                name=name, domain=domain, score=85.0, status="PASS",
                explanation=f"Ensemble consistent (spread={spread:.3f}, avg={avg:.3f}) [static]",
                methodology=methodology + " (static fallback — no tracking data)",
                raw_metrics=raw, thresholds=thresholds,
            )
        if avg > 0:
            return HealthCheckResult(
                name=name, domain=domain, score=55.0, status="WARN",
                explanation=f"Ensemble divergent (spread={spread:.3f}, avg={avg:.3f}) [static]",
                methodology=methodology + " (static fallback — no tracking data)",
                raw_metrics=raw, thresholds=thresholds,
            )
        return HealthCheckResult(
            name=name, domain=domain, score=25.0, status="FAIL",
            explanation=f"Ensemble poor quality (avg holdout_corr={avg:.3f}) [static]",
            methodology=methodology + " (static fallback — no tracking data)",
            raw_metrics=raw, thresholds=thresholds,
        )

    # ── Information Ratio Check (Spec 09) ─────────────────────────────────

    def _check_information_ratio(self) -> HealthCheckResult:
        """Compute rolling Information Ratio and compare to baseline.

        IR = (signal_return - benchmark_return) / tracking_error
        Uses a 20-day rolling window.  Compares current IR to baseline
        (from training period or historical average).

        Score mapping:
            IR >= 1.0  → 90 (excellent alpha generation)
            IR >= 0.5  → 70 (acceptable)
            IR >= 0.0  → 45 (marginal)
            IR <  0.0  → 20 (no alpha)
        """
        domain = "signal_quality"
        name = "information_ratio"
        methodology = (
            "IR = (signal_return - benchmark_return) / tracking_error. "
            "Rolling 20-day window on backtest trade returns vs SPY benchmark. "
            "IR > 1.0 is excellent, 0.5–1.0 acceptable, < 0.5 poor."
        )
        thresholds = {"ir_excellent": 1.0, "ir_acceptable": 0.5, "ir_poor": 0.0}
        try:
            df = self._load_trades_csv()
            if df is None:
                return _unavailable(name, domain, "No backtest trades CSV")

            if "actual_return" not in df.columns:
                return _unavailable(name, domain, "Missing actual_return column")

            actual = pd.to_numeric(df["actual_return"], errors="coerce").dropna()
            if len(actual) < 20:
                return _unavailable(
                    name, domain,
                    f"Only {len(actual)} trade returns — need 20+",
                )

            # Load benchmark returns (SPY) if available
            benchmark_returns = self._load_benchmark_returns()

            # Compute strategy daily-like returns from trade returns
            signal_returns = actual.values.astype(float)

            if benchmark_returns is not None and len(benchmark_returns) >= 20:
                # Align: use as many benchmark returns as we have trade returns
                n = min(len(signal_returns), len(benchmark_returns))
                sig = signal_returns[-n:]
                bench = benchmark_returns[-n:]
            else:
                # Zero benchmark (excess return = signal return)
                sig = signal_returns
                bench = np.zeros_like(sig)

            # Compute rolling IR over 20-day windows
            window = min(20, len(sig))
            if window < 10:
                return _unavailable(
                    name, domain,
                    f"Window {window} too small for IR calculation",
                )

            excess = sig - bench
            rolling_irs = []
            for start in range(0, len(excess) - window + 1, max(1, window // 2)):
                chunk = excess[start:start + window]
                te = float(np.std(chunk, ddof=1))
                if te > 1e-8:
                    ir_val = float(np.mean(chunk)) / te
                    rolling_irs.append(ir_val)

            if not rolling_irs:
                return _unavailable(name, domain, "Could not compute rolling IR")

            current_ir = float(rolling_irs[-1])
            avg_ir = float(np.mean(rolling_irs))

            # Baseline IR: use first third of windows as baseline
            n_baseline = max(1, len(rolling_irs) // 3)
            baseline_ir = float(np.mean(rolling_irs[:n_baseline]))

            # Degradation check
            degradation = None
            if abs(baseline_ir) > 1e-6:
                degradation = current_ir / baseline_ir

            raw = {
                "current_ir": round(current_ir, 4),
                "avg_ir": round(avg_ir, 4),
                "baseline_ir": round(baseline_ir, 4),
                "degradation_ratio": round(degradation, 4) if degradation is not None else None,
                "n_windows": len(rolling_irs),
                "n_trades": len(signal_returns),
                "benchmark_available": benchmark_returns is not None,
            }

            # Score based on current IR level
            if current_ir >= 1.0:
                return HealthCheckResult(
                    name=name, domain=domain, score=90.0, status="PASS",
                    explanation=f"Excellent IR={current_ir:.3f} (baseline={baseline_ir:.3f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if current_ir >= 0.5:
                return HealthCheckResult(
                    name=name, domain=domain, score=70.0, status="PASS",
                    explanation=f"Acceptable IR={current_ir:.3f} (baseline={baseline_ir:.3f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if current_ir >= 0.0:
                return HealthCheckResult(
                    name=name, domain=domain, score=45.0, status="WARN",
                    explanation=f"Marginal IR={current_ir:.3f} (baseline={baseline_ir:.3f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=20.0, status="FAIL",
                explanation=f"Negative IR={current_ir:.3f} — no alpha generation",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    @staticmethod
    def _load_benchmark_returns() -> Optional[np.ndarray]:
        """Load SPY daily returns for benchmark comparison."""
        try:
            from quant_engine.config import DATA_CACHE_DIR, BENCHMARK
            cache_dir = Path(DATA_CACHE_DIR)
            # Try daily cache first, then 4-hour
            for suffix in ["_1d.parquet", "_4hour*.parquet"]:
                paths = sorted(cache_dir.glob(f"{BENCHMARK}{suffix}"))
                if paths:
                    df = pd.read_parquet(paths[0])
                    if "Close" in df.columns:
                        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
                        returns = close.pct_change().dropna().values.astype(float)
                        if len(returns) >= 20:
                            return returns
        except Exception:
            pass
        return None

    # ── Data Quality Checks ─────────────────────────────────────────────

    def _check_survivorship_bias(self) -> HealthCheckResult:
        """Quantified survivorship bias check (Spec 09 enhanced).

        Scans cache for delisted securities and computes the PnL impact
        of positions in those securities as a percentage of total PnL.

        Scoring:
            pnl_loss < 1%  → 90 (minimal bias)
            pnl_loss < 5%  → 55 (moderate bias)
            pnl_loss >= 5% → 25 (significant bias)

        Falls back to binary delisting-return detection when trade data
        is unavailable.
        """
        domain = "data_integrity"
        name = "survivorship_bias"
        methodology = (
            "Identify securities deleted from universe (last bar >1yr old). "
            "Compute PnL from trades in deleted securities as % of total PnL. "
            "pnl_loss < 1% is clean; > 5% indicates significant survivorship bias. "
            "Falls back to delisting return column detection if no trade data."
        )
        thresholds = {"pnl_loss_clean": 0.01, "pnl_loss_moderate": 0.05}
        try:
            from quant_engine.config import DATA_CACHE_DIR
            cache_dir = Path(DATA_CACHE_DIR)
            parquets = sorted(cache_dir.glob("*_1d.parquet"))
            if not parquets:
                return _unavailable(name, domain, "No cached parquet files")

            has_total_ret = 0
            delisted_tickers: List[str] = []
            checked = 0
            cutoff = datetime.now().timestamp() - 365 * 86400  # 1 year ago

            for p in parquets:
                try:
                    df = pd.read_parquet(p)
                    checked += 1
                    if "total_ret" in df.columns:
                        has_total_ret += 1
                    # Check if last bar is old (delisted proxy)
                    if df.index.dtype == "datetime64[ns]" or hasattr(df.index, 'max'):
                        try:
                            last_date = pd.Timestamp(df.index.max())
                            if last_date.timestamp() < cutoff:
                                # Extract ticker from filename
                                ticker = p.stem.replace("_1d", "")
                                delisted_tickers.append(ticker)
                        except (TypeError, ValueError):
                            pass
                except (OSError, ValueError):
                    continue

            if checked == 0:
                return _unavailable(name, domain, "Could not read any cache files")

            # Attempt quantified PnL impact (Spec 09)
            pnl_lost_pct = 0.0
            pnl_quantified = False
            n_deleted_trades = 0

            if delisted_tickers:
                trades_df = self._load_trades_csv()
                if trades_df is not None and "ticker" in trades_df.columns:
                    net_ret = pd.to_numeric(
                        trades_df.get("net_return", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    pos_size = pd.to_numeric(
                        trades_df.get("position_size", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    trade_pnl = (net_ret * pos_size).fillna(0.0)
                    total_pnl = float(trade_pnl.abs().sum())

                    if total_pnl > 0:
                        deleted_mask = trades_df["ticker"].isin(delisted_tickers)
                        deleted_pnl = float(trade_pnl[deleted_mask].sum())
                        n_deleted_trades = int(deleted_mask.sum())
                        pnl_lost_pct = abs(deleted_pnl) / total_pnl
                        pnl_quantified = True

            raw = {
                "files_checked": checked,
                "has_total_ret": has_total_ret,
                "n_delisted": len(delisted_tickers),
                "delisted_tickers": delisted_tickers[:20],
                "pnl_lost_pct": round(pnl_lost_pct, 4),
                "pnl_quantified": pnl_quantified,
                "n_deleted_trades": n_deleted_trades,
            }

            if pnl_quantified:
                # Quantified scoring (Spec 09)
                if pnl_lost_pct < 0.01:
                    return HealthCheckResult(
                        name=name, domain=domain, score=90.0, status="PASS",
                        explanation=(
                            f"Survivorship bias minimal: {pnl_lost_pct:.2%} PnL from "
                            f"{len(delisted_tickers)} delisted securities"
                        ),
                        methodology=methodology, raw_metrics=raw, thresholds=thresholds)
                if pnl_lost_pct < 0.05:
                    return HealthCheckResult(
                        name=name, domain=domain, score=55.0, status="WARN",
                        explanation=(
                            f"Moderate survivorship bias: {pnl_lost_pct:.2%} PnL from "
                            f"{len(delisted_tickers)} delisted securities"
                        ),
                        methodology=methodology, raw_metrics=raw, thresholds=thresholds)
                return HealthCheckResult(
                    name=name, domain=domain, score=25.0, status="FAIL",
                    explanation=(
                        f"Significant survivorship bias: {pnl_lost_pct:.2%} PnL from "
                        f"{len(delisted_tickers)} delisted ({n_deleted_trades} trades)"
                    ),
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)

            # Fallback: binary check (pre-Spec 09 logic)
            total_ret_present = has_total_ret > 0
            if total_ret_present and len(delisted_tickers) > 0:
                return HealthCheckResult(
                    name=name, domain=domain, score=90.0, status="PASS",
                    explanation=f"Delisting returns present ({has_total_ret} files), "
                                f"{len(delisted_tickers)} delisted tickers found",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if total_ret_present or len(delisted_tickers) > 0:
                return HealthCheckResult(
                    name=name, domain=domain, score=60.0, status="WARN",
                    explanation=f"Partial survivorship coverage (total_ret={has_total_ret}, "
                                f"delisted={len(delisted_tickers)})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=30.0, status="FAIL",
                explanation="No delisting returns or delisted tickers — survivorship bias likely",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_data_quality_anomalies(self) -> HealthCheckResult:
        """Systematic anomaly scan across full cached universe.

        Checks zero-volume fraction, extreme daily returns (>40%), and
        stale data (>30 days old) across ALL cached parquets.
        """
        domain = "data_integrity"
        name = "data_anomalies"
        methodology = (
            "Scan ALL cached parquets for: zero-volume fraction >25%, "
            "extreme daily returns >40%, stale data >30 days. "
            "Tickers with any issue are flagged."
        )
        thresholds = {"issue_rate_pass": 0.05, "issue_rate_warn": 0.20}
        try:
            from quant_engine.config import DATA_CACHE_DIR
            cache_dir = Path(DATA_CACHE_DIR)
            parquets = sorted(cache_dir.glob("*_1d.parquet"))
            if not parquets:
                return _unavailable(name, domain, "No cached data files")

            issue_count = 0
            total_checked = 0
            stale_cutoff = datetime.now().timestamp() - 30 * 86400  # 30 days

            for p in parquets:
                try:
                    df = pd.read_parquet(p)
                    total_checked += 1
                    has_issue = False

                    # Zero-volume check
                    if "Volume" in df.columns:
                        vol = pd.to_numeric(df["Volume"], errors="coerce").dropna()
                        if len(vol) > 20 and float((vol == 0).mean()) > 0.25:
                            has_issue = True

                    # Extreme return check
                    if "Close" in df.columns:
                        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
                        if len(close) > 20:
                            daily_ret = close.pct_change().dropna()
                            if (daily_ret.abs() > 0.40).any():
                                has_issue = True

                    # Stale data check
                    if p.stat().st_mtime < stale_cutoff:
                        has_issue = True

                    if has_issue:
                        issue_count += 1
                except (OSError, ValueError):
                    continue

            if total_checked == 0:
                return _unavailable(name, domain, "Could not read any data files")

            issue_rate = issue_count / total_checked
            raw = {"total_checked": total_checked, "issues_found": issue_count,
                   "issue_rate": round(issue_rate, 4)}

            if issue_rate < 0.05:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"{issue_count}/{total_checked} tickers with issues ({issue_rate:.1%})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if issue_rate < 0.20:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"{issue_count}/{total_checked} tickers with issues ({issue_rate:.1%})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"{issue_count}/{total_checked} tickers with issues ({issue_rate:.1%})",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_market_microstructure(self) -> HealthCheckResult:
        """Check position sizes vs liquidity (participation rate).

        Computes position_notional = capital * position_size_pct, compares
        to median daily dollar volume.  Low participation rate means
        positions won't move the market.
        """
        domain = "data_integrity"
        name = "microstructure"
        methodology = (
            "position_notional = BACKTEST_ASSUMED_CAPITAL_USD * POSITION_SIZE_PCT. "
            "participation_rate = position_notional / median_daily_dollar_volume. "
            "<0.5% is excellent, <2% is acceptable."
        )
        thresholds = {"participation_pass": 0.005, "participation_warn": 0.02}
        try:
            from quant_engine.config import (
                DATA_CACHE_DIR, BACKTEST_ASSUMED_CAPITAL_USD, POSITION_SIZE_PCT,
            )
            position_notional = BACKTEST_ASSUMED_CAPITAL_USD * POSITION_SIZE_PCT

            cache_dir = Path(DATA_CACHE_DIR)
            parquets = sorted(cache_dir.glob("*_1d.parquet"))
            if len(parquets) < 3:
                return _unavailable(name, domain, "Insufficient data for microstructure analysis")

            dollar_volumes = []
            for p in parquets:
                try:
                    df = pd.read_parquet(p)
                    if "Volume" in df.columns and "Close" in df.columns:
                        vol = pd.to_numeric(df["Volume"], errors="coerce")
                        close = pd.to_numeric(df["Close"], errors="coerce")
                        dv = (vol * close).dropna().tail(60)
                        if len(dv) > 0:
                            dollar_volumes.append(float(dv.median()))
                except (OSError, ValueError):
                    continue

            if not dollar_volumes:
                return _unavailable(name, domain, "Could not compute dollar volumes")

            median_dv = float(np.median(dollar_volumes))
            if median_dv <= 0:
                return _unavailable(name, domain, "Zero median dollar volume")

            participation = position_notional / median_dv
            raw = {"position_notional": position_notional,
                   "median_dollar_volume": round(median_dv, 0),
                   "participation_rate": round(participation, 6),
                   "tickers_checked": len(dollar_volumes)}

            if participation < 0.005:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"Participation rate {participation:.4%} — minimal market impact",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if participation < 0.02:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"Participation rate {participation:.4%} — moderate impact risk",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"Participation rate {participation:.4%} — high market impact risk",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_wrds_status(self) -> HealthCheckResult:
        """Check whether the WRDS data provider is available.

        WRDS is the primary data source for survivorship-bias-free CRSP prices
        and alternative data (earnings, options, short interest, insider filings).
        When unavailable, the system falls back to local cache which may carry
        survivorship bias and lacks alternative data signals.
        """
        domain = "data_integrity"
        name = "wrds_availability"
        methodology = (
            "Checks whether WRDSProvider.available() returns True, which "
            "requires WRDS_USERNAME env var and a live database connection. "
            "When WRDS is unavailable, OHLCV data comes from local cache "
            "(potential survivorship bias) and alternative data features "
            "(earnings, options, short interest, insider) are disabled."
        )
        try:
            from quant_engine.config import WRDS_ENABLED
            if not WRDS_ENABLED:
                return HealthCheckResult(
                    name=name, domain=domain, score=50.0, status="WARN",
                    explanation="WRDS_ENABLED=False — operating on cached/local data only",
                    methodology=methodology,
                    raw_metrics={"wrds_enabled": False, "wrds_connected": False},
                )

            from quant_engine.data.wrds_provider import WRDSProvider
            provider = WRDSProvider()
            if provider.available():
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation="WRDS connected — survivorship-bias-free data active",
                    methodology=methodology,
                    raw_metrics={"wrds_enabled": True, "wrds_connected": True},
                )
            return HealthCheckResult(
                name=name, domain=domain, score=30.0, status="FAIL",
                explanation=(
                    "WRDS_ENABLED=True but connection unavailable — "
                    "operating on cached/local data only"
                ),
                methodology=methodology,
                raw_metrics={"wrds_enabled": True, "wrds_connected": False},
            )
        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    # ── Risk Management Checks ──────────────────────────────────────────

    def _check_tail_risk(self) -> HealthCheckResult:
        """Compute CVaR and max drawdown from backtest trade-level returns.

        Uses backtest_10d_trades.csv (not equity CSV which doesn't exist).
        Reconstructs daily P&L from trade returns and position sizes.
        """
        domain = "risk_management"
        name = "tail_risk"
        methodology = (
            "Reconstruct portfolio returns from backtest trades (net_return * position_size). "
            "Compute CVaR at 95th percentile and max drawdown. "
            "CVaR > -3% AND reasonable max drawdown indicates controlled tail risk."
        )
        thresholds = {"cvar95_pass": -0.03, "cvar95_warn": -0.05}
        try:
            df = self._load_trades_csv()
            if df is None:
                return _unavailable(name, domain, "No backtest trades CSV")

            if "net_return" not in df.columns or "position_size" not in df.columns:
                return _unavailable(name, domain, "Trades CSV missing net_return/position_size")

            # Compute trade-level portfolio contribution
            trade_returns = (
                pd.to_numeric(df["net_return"], errors="coerce")
                * pd.to_numeric(df["position_size"], errors="coerce")
            ).dropna()

            if len(trade_returns) < 30:
                return _unavailable(name, domain, f"Only {len(trade_returns)} trade returns — need 30+")

            returns = trade_returns.values.astype(float)
            var5 = float(np.percentile(returns, 5))
            tail = returns[returns <= var5]
            cvar5 = float(tail.mean()) if len(tail) > 0 else var5

            # Max drawdown from cumulative returns
            cum = np.cumsum(returns)
            running_max = np.maximum.accumulate(cum)
            drawdowns = cum - running_max
            max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

            raw = {"cvar_95": round(cvar5, 4), "var_5": round(var5, 4),
                   "max_drawdown": round(max_dd, 4), "n_trades": len(returns)}

            if cvar5 > -0.03 and max_dd > -0.15:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"CVaR95={cvar5:.4f}, MaxDD={max_dd:.4f} — controlled tail risk",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if cvar5 > -0.05:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"CVaR95={cvar5:.4f}, MaxDD={max_dd:.4f} — elevated tail risk",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"CVaR95={cvar5:.4f}, MaxDD={max_dd:.4f} — extreme tail risk",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_correlation_regime(self) -> HealthCheckResult:
        """Check pairwise correlation of actual portfolio holdings.

        Tries paper trader positions first; if empty, falls back to
        backtest trades (groups concurrent trades, computes pairwise
        correlation of held tickers from cache parquets).
        """
        domain = "risk_management"
        name = "correlation_regime"
        methodology = (
            "Identify held tickers from paper trader or backtest trades. "
            "Load daily returns from cache, compute avg pairwise correlation. "
            "<0.40 indicates good diversification."
        )
        thresholds = {"avg_corr_pass": 0.40, "avg_corr_warn": 0.65}
        try:
            from quant_engine.config import DATA_CACHE_DIR, RESULTS_DIR

            # Try paper trader positions first
            tickers_to_check = []
            state_path = Path(RESULTS_DIR) / "autopilot" / "paper_state.json"
            if state_path.exists():
                with open(state_path, "r") as f:
                    state = json.load(f)
                positions = state.get("positions", [])
                tickers_to_check = [p.get("ticker", "") for p in positions if p.get("ticker")]

            # Fall back to backtest trades
            if len(tickers_to_check) < 3:
                df = self._load_trades_csv()
                if df is not None and "ticker" in df.columns:
                    # Use the most frequently traded tickers
                    top_tickers = df["ticker"].value_counts().head(20).index.tolist()
                    tickers_to_check = top_tickers

            if len(tickers_to_check) < 3:
                return _unavailable(name, domain, "Not enough tickers for correlation check")

            # Load returns from cache
            cache_dir = Path(DATA_CACHE_DIR)
            returns_list = []
            for ticker in tickers_to_check:
                p = cache_dir / f"{ticker}_1d.parquet"
                if not p.exists():
                    continue
                try:
                    df_t = pd.read_parquet(p)
                    if "Close" in df_t.columns:
                        ret = pd.to_numeric(df_t["Close"], errors="coerce").pct_change().dropna()
                        ret.name = ticker
                        returns_list.append(ret.tail(252))
                except (OSError, ValueError):
                    continue

            if len(returns_list) < 3:
                return _unavailable(name, domain, "Could not load enough return series")

            ret_df = pd.concat(returns_list, axis=1).dropna()
            if ret_df.shape[0] < 20 or ret_df.shape[1] < 3:
                return _unavailable(name, domain, "Insufficient overlapping return data")

            corr_matrix = ret_df.corr()
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = float(corr_matrix.values[mask].mean())
            raw = {"avg_pairwise_corr": round(avg_corr, 4),
                   "n_assets": ret_df.shape[1], "n_days": ret_df.shape[0],
                   "source": "paper_trader" if len(tickers_to_check) > 0 else "backtest"}

            if avg_corr < 0.40:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"Avg pairwise correlation: {avg_corr:.3f} (diversified)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if avg_corr < 0.65:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"Avg pairwise correlation: {avg_corr:.3f} (elevated)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"Avg pairwise correlation: {avg_corr:.3f} (crisis-level)",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_capital_utilization(self) -> HealthCheckResult:
        """Check capital deployment with Herfindahl diversification index.

        Loads paper state positions, computes utilization ratio AND HHI
        of position sizes to detect over-concentration.
        """
        domain = "risk_management"
        name = "capital_utilization"
        methodology = (
            "Load paper trader positions.  Compute utilization ratio "
            "(invested / equity) and HHI of position weights. "
            "Low HHI + moderate utilization is ideal."
        )
        thresholds = {"utilization_low": 0.10, "utilization_high": 0.95,
                      "hhi_pass": 0.15}
        try:
            from quant_engine.config import RESULTS_DIR, PAPER_INITIAL_CAPITAL
            state_path = Path(RESULTS_DIR) / "autopilot" / "paper_state.json"
            if not state_path.exists():
                return _unavailable(name, domain, "No paper trading state")

            with open(state_path, "r") as f:
                state = json.load(f)

            positions = state.get("positions", [])
            cash = float(state.get("cash", PAPER_INITIAL_CAPITAL))
            equity = float(state.get("equity", cash))
            if equity <= 0:
                equity = cash if cash > 0 else PAPER_INITIAL_CAPITAL

            if not positions:
                # No positions — report utilization as 0%
                raw = {"utilization": 0.0, "hhi": 0.0, "n_positions": 0,
                       "equity": equity, "cash": cash}
                return HealthCheckResult(
                    name=name, domain=domain, score=50.0, status="WARN",
                    explanation="No open positions — capital fully idle",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)

            position_values = [abs(float(p.get("value", 0))) for p in positions]
            total_invested = sum(position_values)
            utilization = total_invested / equity if equity > 0 else 0.0

            # Herfindahl index (HHI) of position weights
            if total_invested > 0:
                weights = [v / total_invested for v in position_values]
                hhi = float(sum(w ** 2 for w in weights))
            else:
                hhi = 0.0

            raw = {"utilization": round(utilization, 4), "hhi": round(hhi, 4),
                   "n_positions": len(positions), "equity": equity,
                   "total_invested": round(total_invested, 2)}

            if 0.30 <= utilization <= 0.90 and hhi < 0.15:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"Utilization {utilization:.0%}, HHI={hhi:.3f} (diversified)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if utilization < 0.10:
                return HealthCheckResult(
                    name=name, domain=domain, score=50.0, status="WARN",
                    explanation=f"Utilization {utilization:.0%} (underdeployed)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if utilization > 0.95 or hhi > 0.25:
                return HealthCheckResult(
                    name=name, domain=domain, score=45.0, status="WARN",
                    explanation=f"Utilization {utilization:.0%}, HHI={hhi:.3f} (concentrated)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=70.0, status="PASS",
                explanation=f"Utilization {utilization:.0%}, HHI={hhi:.3f}",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    # ── Execution Quality Checks ────────────────────────────────────────

    def _check_execution_quality(self) -> HealthCheckResult:
        """Check execution model accuracy using backtest TCA data.

        If TCA columns present (fill_ratio, entry_impact_bps, exit_impact_bps),
        analyze them directly.  Otherwise, use predicted vs actual return
        correlation as a proxy for execution model accuracy.
        """
        domain = "execution_quality"
        name = "execution_quality"
        methodology = (
            "Load backtest trades.  If TCA columns present: analyze fill ratio "
            "and impact.  Otherwise: use predicted-vs-actual return correlation "
            "as proxy for execution model accuracy."
        )
        try:
            df = self._load_trades_csv()
            if df is None:
                return _unavailable(name, domain, "No backtest trades CSV")

            # Check for TCA columns
            tca_cols = {"fill_ratio", "entry_impact_bps", "exit_impact_bps"}
            has_tca = tca_cols.issubset(set(df.columns))

            if has_tca:
                fill_ratio = pd.to_numeric(df["fill_ratio"], errors="coerce").dropna()
                entry_impact = pd.to_numeric(df["entry_impact_bps"], errors="coerce").dropna()
                exit_impact = pd.to_numeric(df["exit_impact_bps"], errors="coerce").dropna()

                avg_fill = float(fill_ratio.mean()) if len(fill_ratio) > 0 else 0.0
                avg_impact = float((entry_impact.mean() + exit_impact.mean()) / 2) if len(entry_impact) > 0 else 0.0

                raw = {"avg_fill_ratio": round(avg_fill, 4),
                       "avg_impact_bps": round(avg_impact, 2),
                       "data_source": "TCA", "n_trades": len(df)}
                thresholds = {"fill_ratio_pass": 0.90, "impact_pass": 10.0}

                if avg_fill > 0.90 and avg_impact < 10.0:
                    return HealthCheckResult(
                        name=name, domain=domain, score=85.0, status="PASS",
                        explanation=f"Fill={avg_fill:.1%}, avg impact={avg_impact:.1f}bps",
                        methodology=methodology, raw_metrics=raw, thresholds=thresholds)
                if avg_fill > 0.70:
                    return HealthCheckResult(
                        name=name, domain=domain, score=55.0, status="WARN",
                        explanation=f"Fill={avg_fill:.1%}, avg impact={avg_impact:.1f}bps",
                        methodology=methodology, raw_metrics=raw, thresholds=thresholds)
                return HealthCheckResult(
                    name=name, domain=domain, score=25.0, status="FAIL",
                    explanation=f"Poor execution: Fill={avg_fill:.1%}, impact={avg_impact:.1f}bps",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)

            # No TCA — use prediction-realization correlation as proxy
            if "predicted_return" not in df.columns or "actual_return" not in df.columns:
                return _unavailable(name, domain, "No TCA columns and no predicted/actual returns")

            pred = pd.to_numeric(df["predicted_return"], errors="coerce")
            actual = pd.to_numeric(df["actual_return"], errors="coerce")
            valid = pred.notna() & actual.notna()
            if valid.sum() < 20:
                return _unavailable(name, domain, "Too few prediction-actual pairs")

            corr, _ = sp_stats.spearmanr(pred[valid], actual[valid])
            # Slippage proxy: median absolute difference between predicted and net return
            if "net_return" in df.columns:
                net = pd.to_numeric(df["net_return"], errors="coerce")
                slip_proxy = float((actual[valid] - net[valid]).abs().median())
            else:
                slip_proxy = 0.0

            raw = {"pred_actual_corr": round(float(corr), 4),
                   "median_slippage_proxy": round(slip_proxy, 6),
                   "data_source": "prediction_proxy", "n_pairs": int(valid.sum())}
            thresholds = {"corr_pass": 0.05, "corr_warn": 0.0}

            if corr > 0.05:
                return HealthCheckResult(
                    name=name, domain=domain, score=75.0, status="PASS",
                    explanation=f"Pred-actual correlation={corr:.4f} (no TCA; proxy only)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if corr > 0.0:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"Weak pred-actual correlation={corr:.4f} (no TCA; proxy only)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=30.0, status="FAIL",
                explanation=f"Negative pred-actual correlation={corr:.4f}",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_signal_profitability(self) -> HealthCheckResult:
        """Check if long signals are actually profitable (renamed from information_ratio).

        Uses trades CSV: compute average return of long-signal trades and
        overall IC (Spearman of predicted vs actual).
        """
        domain = "execution_quality"
        name = "signal_profitability"
        methodology = (
            "From backtest trades: compute average return of trades where "
            "predicted_return > 0 (long signal) and overall IC. "
            "Both positive = signals have real edge."
        )
        thresholds = {"long_return_pass": 0.0, "ic_pass": 0.0}
        try:
            df = self._load_trades_csv()
            if df is None:
                return _unavailable(name, domain, "No backtest trades CSV")
            if "predicted_return" not in df.columns or "actual_return" not in df.columns:
                return _unavailable(name, domain, "Missing predicted/actual columns")

            pred = pd.to_numeric(df["predicted_return"], errors="coerce")
            actual = pd.to_numeric(df["actual_return"], errors="coerce")
            valid = pred.notna() & actual.notna()
            if valid.sum() < 20:
                return _unavailable(name, domain, "Too few trades")

            long_mask = pred[valid] > 0
            long_return = float(actual[valid][long_mask].mean()) if long_mask.sum() > 0 else 0.0
            ic, _ = sp_stats.spearmanr(pred[valid], actual[valid])

            raw = {"long_avg_return": round(long_return, 6),
                   "ic_spearman": round(float(ic), 4),
                   "n_long_trades": int(long_mask.sum()),
                   "n_total_trades": int(valid.sum())}

            if long_return > 0 and ic > 0:
                score = min(90.0, 70.0 + ic * 200)  # scale IC contribution
                return HealthCheckResult(
                    name=name, domain=domain, score=score, status="PASS",
                    explanation=f"Long signal avg return={long_return:.4f}, IC={ic:.4f}",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if long_return > 0 or ic > 0:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"Mixed signal: long_ret={long_return:.4f}, IC={ic:.4f}",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"No edge: long_ret={long_return:.4f}, IC={ic:.4f}",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    # ── Model Governance Checks ─────────────────────────────────────────

    def _check_cv_gap_trend(self) -> HealthCheckResult:
        """Detect overfitting from CV gap — with absolute gap threshold.

        Gap > 0.15 always FAIL regardless of trend.
        gap < 0.05 AND trend < 0 → PASS(85).
        """
        domain = "model_governance"
        name = "cv_gap_trend"
        methodology = (
            "Load model registry versions.  Check latest CV gap against "
            "absolute threshold (0.15 = always FAIL).  Compute trend slope "
            "across versions.  gap < 0.05 with decreasing trend is ideal."
        )
        thresholds = {"gap_fail": 0.15, "gap_pass": 0.05, "trend_pass": 0.0}
        try:
            from quant_engine.config import MODEL_DIR
            registry_path = Path(MODEL_DIR) / "registry.json"
            if not registry_path.exists():
                return _unavailable(name, domain, "No model registry")

            with open(registry_path, "r") as f:
                reg = json.load(f)

            versions = reg.get("versions", []) if isinstance(reg, dict) else []
            gaps = [float(v.get("cv_gap", 0)) for v in versions if "cv_gap" in v]

            if not gaps:
                return _unavailable(name, domain, "No CV gap data in registry")

            latest_gap = gaps[-1]

            # Compute trend if 2+ versions
            if len(gaps) >= 2:
                x = np.arange(len(gaps), dtype=float)
                slope = float(np.polyfit(x, gaps, 1)[0])
            else:
                slope = 0.0

            raw = {"latest_gap": round(latest_gap, 4), "trend_slope": round(slope, 6),
                   "n_versions": len(gaps), "all_gaps": [round(g, 4) for g in gaps]}

            # Absolute threshold: gap > 0.15 is always FAIL
            if latest_gap > 0.15:
                return HealthCheckResult(
                    name=name, domain=domain, score=20.0, status="FAIL",
                    explanation=f"CV gap {latest_gap:.4f} exceeds 0.15 — severe overfitting",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)

            if latest_gap < 0.05 and slope <= 0:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"CV gap {latest_gap:.4f}, trend {slope:+.4f}/version (stable)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if latest_gap < 0.10:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"CV gap {latest_gap:.4f}, trend {slope:+.4f}/version",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=30.0, status="FAIL",
                explanation=f"CV gap {latest_gap:.4f}, trend {slope:+.4f}/version (overfitting risk)",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_retraining_effectiveness(self) -> HealthCheckResult:
        """Track slope direction of holdout Spearman across model versions.

        Improving slope → PASS(85), stable → WARN(65), declining → FAIL(25).
        """
        domain = "model_governance"
        name = "retrain_effectiveness"
        methodology = (
            "Compute linear regression slope of holdout Spearman values "
            "across model versions.  Positive slope indicates retrains are "
            "improving model quality."
        )
        thresholds = {"slope_pass": 0.0, "slope_warn": -0.005}
        try:
            from quant_engine.config import MODEL_DIR
            registry_path = Path(MODEL_DIR) / "registry.json"
            if not registry_path.exists():
                return _unavailable(name, domain, "No model registry")

            with open(registry_path, "r") as f:
                reg = json.load(f)

            versions = reg.get("versions", []) if isinstance(reg, dict) else []
            spearman_vals = [float(v.get("holdout_spearman", 0)) for v in versions
                            if "holdout_spearman" in v]

            if len(spearman_vals) < 2:
                return _unavailable(name, domain, f"Need 2+ versions (have {len(spearman_vals)})")

            x = np.arange(len(spearman_vals), dtype=float)
            slope = float(np.polyfit(x, spearman_vals, 1)[0])
            latest = spearman_vals[-1]
            raw = {"slope": round(slope, 6), "latest_spearman": round(latest, 4),
                   "n_versions": len(spearman_vals),
                   "all_values": [round(v, 4) for v in spearman_vals]}

            if slope > 0:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"Retrains improving (slope={slope:+.4f}, latest={latest:.4f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if slope > -0.005:
                return HealthCheckResult(
                    name=name, domain=domain, score=65.0, status="WARN",
                    explanation=f"Retrains stable (slope={slope:+.4f}, latest={latest:.4f})",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"Retrains declining (slope={slope:+.4f}, latest={latest:.4f})",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    def _check_feature_importance_drift(self) -> HealthCheckResult:
        """Check feature importance drift with top-3 rank change explanation."""
        domain = "model_governance"
        name = "feature_drift"
        methodology = (
            "Load FeatureStabilityTracker, get Spearman rank correlation "
            "between latest and previous training cycles.  Identify top-3 "
            "features with largest rank changes for explanation."
        )
        try:
            from quant_engine.models.feature_stability import FeatureStabilityTracker
            tracker = FeatureStabilityTracker()
            report = tracker.check_stability()

            if report.n_cycles < 2:
                return _unavailable(name, domain, f"Only {report.n_cycles} training cycles")

            spearman = report.spearman_vs_previous
            if spearman is None:
                return _unavailable(name, domain, "No comparison available")

            raw: Dict[str, Any] = {"spearman_vs_previous": round(spearman, 4),
                   "n_cycles": report.n_cycles}

            # Compute top-3 rank changes if we have the data
            rank_change_explanation = ""
            try:
                history_path = Path("results/feature_stability_history.json")
                if history_path.exists():
                    with open(history_path) as f:
                        hist = json.load(f)
                    cycles = hist.get("cycles", [])
                    if len(cycles) >= 2:
                        current = cycles[-1].get("top_features", [])
                        previous = cycles[-2].get("top_features", [])
                        if current and previous:
                            changes = []
                            for feat in set(current + previous):
                                cur_rank = current.index(feat) + 1 if feat in current else len(current) + 1
                                prev_rank = previous.index(feat) + 1 if feat in previous else len(previous) + 1
                                changes.append((feat, prev_rank - cur_rank))  # positive = improved
                            changes.sort(key=lambda x: abs(x[1]), reverse=True)
                            top3 = changes[:3]
                            rank_change_explanation = "Top 3 rank changes: " + ", ".join(
                                f"{f} ({d:+d})" for f, d in top3
                            )
                            raw["top_rank_changes"] = {f: d for f, d in top3}
            except Exception:
                pass

            thresholds = {"spearman_pass": 0.70, "spearman_warn": 0.40}
            explanation_parts = [f"Feature rank correlation: {spearman:.3f}"]
            if rank_change_explanation:
                explanation_parts.append(rank_change_explanation)

            if spearman > 0.70:
                return HealthCheckResult(
                    name=name, domain=domain, score=90.0, status="PASS",
                    explanation=" | ".join(explanation_parts) + " (stable)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            if spearman > 0.40:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=" | ".join(explanation_parts) + " (moderate drift)",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)
            return HealthCheckResult(
                name=name, domain=domain, score=20.0, status="FAIL",
                explanation=" | ".join(explanation_parts) + " (severe drift)",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    # ── Alert Integration (Spec 09) ────────────────────────────────────

    def _run_alerts(
        self,
        overall_score: Optional[float],
        domains: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run alert checks and return any triggered alerts.

        Called during compute_comprehensive_health().  Compares today's
        health to yesterday's (from history) for degradation detection.
        """
        try:
            from .health_alerts import create_alert_manager

            alert_mgr = create_alert_manager()
            pending_alerts = []

            # Check domain failures
            domain_scores = {}
            for dname, dinfo in domains.items():
                score = dinfo.get("score")
                if score is not None:
                    domain_scores[dname] = score
            domain_alerts = alert_mgr.check_domain_failures(domain_scores)
            pending_alerts.extend(domain_alerts)

            # Check day-over-day degradation using history
            if overall_score is not None:
                history = self.get_health_history(limit=2)
                if history:
                    yesterday_score = history[-1].get("overall_score")
                    if yesterday_score is not None:
                        deg_alert = alert_mgr.check_health_degradation(
                            overall_score, yesterday_score,
                        )
                        if deg_alert is not None:
                            pending_alerts.append(deg_alert)

            # Process (deduplicate + dispatch)
            sent = alert_mgr.process_alerts(pending_alerts)
            return [a.to_dict() for a in sent]

        except Exception as e:
            logger.warning("Alert processing failed: %s", e)
            return []

    # ── Health History Storage ─────────────────────────────────────────

    _HEALTH_DB_PATH: Optional[Path] = None
    _MAX_SNAPSHOTS = 90  # Spec 09: 90 days retention (up from 30)

    @classmethod
    def _get_health_db_path(cls) -> Path:
        """Return the path to the health history SQLite database."""
        if cls._HEALTH_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._HEALTH_DB_PATH = Path(RESULTS_DIR) / "health_history.db"
        return cls._HEALTH_DB_PATH

    @classmethod
    def _ensure_health_table(cls) -> None:
        """Create health_history table if it doesn't exist."""
        import sqlite3
        db_path = cls._get_health_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_score REAL,
                    domain_scores TEXT,
                    check_results TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_health_snapshot(self, health_result: Dict[str, Any]) -> None:
        """Save a health check snapshot to the SQLite database."""
        import sqlite3
        try:
            self._ensure_health_table()
            db_path = self._get_health_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                timestamp = datetime.now(timezone.utc).isoformat()
                overall_score = health_result.get("overall_score")
                domain_scores = json.dumps({
                    k: {"score": v.get("score"), "status": v.get("status")}
                    for k, v in health_result.get("domains", {}).items()
                })
                check_results = json.dumps(health_result)

                conn.execute(
                    "INSERT INTO health_history (timestamp, overall_score, domain_scores, check_results) "
                    "VALUES (?, ?, ?, ?)",
                    (timestamp, overall_score, domain_scores, check_results),
                )

                # Prune old entries beyond _MAX_SNAPSHOTS
                conn.execute(
                    "DELETE FROM health_history WHERE id NOT IN "
                    "(SELECT id FROM health_history ORDER BY id DESC LIMIT ?)",
                    (self._MAX_SNAPSHOTS,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to save health snapshot: %s", e)

    def get_health_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieve recent health snapshots for trend visualization.

        Spec 09 enhancements:
            - 7-day and 30-day rolling averages
            - Trend detection (improving / stable / degrading)
        """
        import sqlite3
        try:
            self._ensure_health_table()
            db_path = self._get_health_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.execute(
                    "SELECT timestamp, overall_score, domain_scores "
                    "FROM health_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                history = []
                for ts, score, domain_json in reversed(rows):
                    entry: Dict[str, Any] = {
                        "timestamp": ts,
                        "overall_score": score,
                    }
                    try:
                        entry["domain_scores"] = json.loads(domain_json)
                    except (json.JSONDecodeError, TypeError):
                        entry["domain_scores"] = {}
                    history.append(entry)
                return history
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to load health history: %s", e)
            return []

    def get_health_history_with_trends(
        self,
        limit: int = 90,
    ) -> Dict[str, Any]:
        """Retrieve health history with rolling averages and trend analysis.

        Returns
        -------
        dict with keys:
            snapshots: list of {timestamp, overall_score, domain_scores}
            rolling_7d: list of 7-day rolling averages
            rolling_30d: list of 30-day rolling averages
            trend: "improving", "stable", or "degrading"
            trend_slope: linear regression slope per day
        """
        history = self.get_health_history(limit=limit)

        if not history:
            return {
                "snapshots": [],
                "rolling_7d": [],
                "rolling_30d": [],
                "trend": "unknown",
                "trend_slope": 0.0,
            }

        scores = [h.get("overall_score") for h in history]
        # Filter out None values for numeric operations
        valid_scores = [s for s in scores if s is not None]

        # Rolling averages
        rolling_7d = self._compute_rolling_average(valid_scores, window=7)
        rolling_30d = self._compute_rolling_average(valid_scores, window=30)

        # Trend detection using linear regression on recent 30 points
        trend, slope = self._detect_trend(valid_scores, window=30)

        return {
            "snapshots": history,
            "rolling_7d": [round(v, 1) for v in rolling_7d],
            "rolling_30d": [round(v, 1) for v in rolling_30d],
            "trend": trend,
            "trend_slope": round(slope, 4),
        }

    @staticmethod
    def _compute_rolling_average(
        scores: List[float],
        window: int = 7,
    ) -> List[float]:
        """Compute rolling average of health scores."""
        if not scores:
            return []
        arr = np.array(scores, dtype=float)
        if len(arr) < window:
            window = len(arr)
        if window <= 0:
            return scores
        cumsum = np.cumsum(arr)
        cumsum = np.insert(cumsum, 0, 0.0)
        rolling = (cumsum[window:] - cumsum[:-window]) / window
        # Pad front with partial averages
        pad = [float(np.mean(arr[:i + 1])) for i in range(min(window - 1, len(arr)))]
        return pad + rolling.tolist()

    @staticmethod
    def _detect_trend(
        scores: List[float],
        window: int = 30,
    ) -> tuple:
        """Detect health score trend using linear regression.

        Returns
        -------
        (trend_label, slope)
            trend_label: "improving" (slope > 0.5), "degrading" (slope < -0.5),
                         or "stable" (|slope| <= 0.5).
            slope: linear regression slope per observation (on 0–100 scale).
        """
        if len(scores) < 3:
            return ("unknown", 0.0)

        recent = scores[-window:] if len(scores) >= window else scores
        x = np.arange(len(recent), dtype=float)
        coeffs = np.polyfit(x, recent, 1)
        slope = float(coeffs[0])

        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "degrading"
        else:
            trend = "stable"

        return (trend, slope)

    def _check_regime_transition_health(self) -> HealthCheckResult:
        """Check regime model quality via entropy, transition frequency,
        and predictive power (replaces HMM diagonal check).

        Computes Shannon entropy of regime distribution and checks if
        regime predicts volatility changes.
        """
        domain = "model_governance"
        name = "regime_transitions"
        methodology = (
            "Compute Shannon entropy of regime probability distribution "
            "from prob_history.  Check correlation between regime and "
            "realized volatility.  Entropy > 0.5 AND predictive_corr > 0.2 "
            "indicates a useful regime model."
        )
        thresholds = {"entropy_pass": 0.5, "predictive_corr_pass": 0.2}
        try:
            from quant_engine.config import DATA_CACHE_DIR
            from .data_helpers import compute_regime_payload

            regime = compute_regime_payload(Path(DATA_CACHE_DIR))
            if regime is None:
                return _unavailable(name, domain, "No regime data available")

            prob_history = regime.get("prob_history", None)
            current_probs = regime.get("current_probs", None)

            # Compute entropy from current regime probabilities
            entropy = 0.0
            if current_probs and isinstance(current_probs, dict):
                probs = [float(v) for v in current_probs.values() if float(v) > 0]
                if probs:
                    probs = np.array(probs)
                    probs = probs / probs.sum()  # normalize
                    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
            elif isinstance(prob_history, pd.DataFrame) and not prob_history.empty:
                # Use average probabilities across time
                prob_cols = [c for c in prob_history.columns if c.startswith("regime_prob_")]
                if prob_cols:
                    avg_probs = prob_history[prob_cols].mean().values
                    avg_probs = avg_probs[avg_probs > 0]
                    if len(avg_probs) > 0:
                        avg_probs = avg_probs / avg_probs.sum()
                        entropy = float(-np.sum(avg_probs * np.log2(avg_probs + 1e-10)))

            # Check predictive power: does regime correlate with realized vol?
            predictive_corr = 0.0
            if isinstance(prob_history, pd.DataFrame) and not prob_history.empty:
                prob_cols = [c for c in prob_history.columns if c.startswith("regime_prob_")]
                if prob_cols and len(prob_history) > 30:
                    # Use dominant regime as signal
                    dominant = prob_history[prob_cols].idxmax(axis=1).astype("category").cat.codes
                    # Try to get realized vol from SPY cache
                    try:
                        spy_path = Path(DATA_CACHE_DIR) / "SPY_1d.parquet"
                        if spy_path.exists():
                            spy = pd.read_parquet(spy_path)
                            if "Close" in spy.columns:
                                spy_ret = pd.to_numeric(spy["Close"], errors="coerce").pct_change()
                                rvol = spy_ret.rolling(20).std()
                                # Align indices
                                common = dominant.index.intersection(rvol.index)
                                if len(common) > 30:
                                    corr, _ = sp_stats.spearmanr(
                                        dominant.reindex(common).dropna(),
                                        rvol.reindex(common).dropna(),
                                    )
                                    if not np.isnan(corr):
                                        predictive_corr = abs(float(corr))
                    except Exception:
                        pass

            raw = {"entropy": round(entropy, 4),
                   "predictive_corr": round(predictive_corr, 4)}

            if entropy > 0.5 and predictive_corr > 0.2:
                return HealthCheckResult(
                    name=name, domain=domain, score=85.0, status="PASS",
                    explanation=f"Entropy={entropy:.3f}, vol predictive corr={predictive_corr:.3f}",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)

            if entropy > 0.3 or predictive_corr > 0.1:
                return HealthCheckResult(
                    name=name, domain=domain, score=55.0, status="WARN",
                    explanation=f"Entropy={entropy:.3f}, vol predictive corr={predictive_corr:.3f} — partial",
                    methodology=methodology, raw_metrics=raw, thresholds=thresholds)

            return HealthCheckResult(
                name=name, domain=domain, score=25.0, status="FAIL",
                explanation=f"Entropy={entropy:.3f}, vol predictive corr={predictive_corr:.3f} — poor regime model",
                methodology=methodology, raw_metrics=raw, thresholds=thresholds)

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    # ── IC Tracking (SPEC-H01) ─────────────────────────────────────────

    _IC_DB_PATH: Optional[Path] = None
    _MAX_IC_SNAPSHOTS = 200  # Retain up to 200 autopilot cycle IC records

    @classmethod
    def _get_ic_db_path(cls) -> Path:
        """Return the path to the IC tracking SQLite database."""
        if cls._IC_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._IC_DB_PATH = Path(RESULTS_DIR) / "ic_tracking.db"
        return cls._IC_DB_PATH

    @classmethod
    def _ensure_ic_table(cls) -> None:
        """Create ic_history table if it doesn't exist."""
        import sqlite3
        db_path = cls._get_ic_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ic_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ic_mean REAL NOT NULL,
                    ic_ir REAL,
                    n_candidates INTEGER,
                    n_passed INTEGER,
                    best_strategy_id TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_ic_snapshot(
        self,
        ic_mean: float,
        ic_ir: Optional[float] = None,
        n_candidates: int = 0,
        n_passed: int = 0,
        best_strategy_id: str = "",
    ) -> None:
        """Save IC metrics from an autopilot cycle to the tracking database.

        Called by the autopilot engine after evaluating candidates.
        """
        import sqlite3
        try:
            self._ensure_ic_table()
            db_path = self._get_ic_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                timestamp = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO ic_history "
                    "(timestamp, ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (timestamp, ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id),
                )
                # Prune old entries
                conn.execute(
                    "DELETE FROM ic_history WHERE id NOT IN "
                    "(SELECT id FROM ic_history ORDER BY id DESC LIMIT ?)",
                    (self._MAX_IC_SNAPSHOTS,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to save IC snapshot: %s", e)

    def get_ic_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent IC snapshots for tracking and health checks.

        Parameters
        ----------
        limit : int
            Maximum number of recent snapshots to return.

        Returns
        -------
        List of dicts, each with: timestamp, ic_mean, ic_ir,
        n_candidates, n_passed, best_strategy_id.
        Ordered chronologically (oldest first).
        """
        import sqlite3
        try:
            self._ensure_ic_table()
            db_path = self._get_ic_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.execute(
                    "SELECT timestamp, ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id "
                    "FROM ic_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                history = []
                for ts, ic_mean, ic_ir, n_cand, n_pass, strat_id in reversed(rows):
                    history.append({
                        "timestamp": ts,
                        "ic_mean": ic_mean,
                        "ic_ir": ic_ir,
                        "n_candidates": n_cand,
                        "n_passed": n_pass,
                        "best_strategy_id": strat_id,
                    })
                return history
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to load IC history: %s", e)
            return []

    def _check_ic_tracking(self) -> HealthCheckResult:
        """Track rolling IC across autopilot cycles (SPEC-H01).

        Retrieves last N autopilot cycle IC values, computes rolling mean
        and trend, and flags:
            - WARNING if rolling IC mean < 0.01
            - CRITICAL (FAIL) if rolling IC mean < 0

        Falls back to extracting IC from the latest autopilot cycle report
        if no IC history database exists yet.

        Score mapping:
            IC mean >= 0.03  → 90 (strong predictive signal)
            IC mean >= 0.01  → 70 (acceptable signal)
            IC mean >= 0.00  → 40 (marginal — WARNING)
            IC mean <  0.00  → 15 (negative IC — CRITICAL)
        """
        domain = "signal_quality"
        name = "ic_tracking"
        methodology = (
            "Retrieve IC (Information Coefficient) values from last N autopilot "
            "cycles.  Compute rolling mean and linear trend.  IC is the Spearman "
            "rank correlation between predicted and actual returns — values > 0.01 "
            "indicate useful predictive signal.  IC < 0 means the model is "
            "anti-predictive."
        )
        try:
            from quant_engine.config import (
                IC_TRACKING_LOOKBACK,
                IC_TRACKING_WARN_THRESHOLD,
                IC_TRACKING_CRITICAL_THRESHOLD,
            )
        except ImportError:
            IC_TRACKING_LOOKBACK = 20
            IC_TRACKING_WARN_THRESHOLD = 0.01
            IC_TRACKING_CRITICAL_THRESHOLD = 0.0

        thresholds = {
            "ic_strong": 0.03,
            "ic_warn": IC_TRACKING_WARN_THRESHOLD,
            "ic_critical": IC_TRACKING_CRITICAL_THRESHOLD,
        }

        try:
            # Primary: read from IC history database
            ic_history = self.get_ic_history(limit=IC_TRACKING_LOOKBACK)
            ic_values = [h["ic_mean"] for h in ic_history if h["ic_mean"] is not None]

            # Fallback: extract from latest autopilot cycle report if no history
            if not ic_values:
                ic_from_report = self._extract_ic_from_cycle_report()
                if ic_from_report is not None:
                    ic_values = [ic_from_report]

            if not ic_values:
                return _unavailable(
                    name, domain,
                    "No IC data available — run at least one autopilot cycle",
                )

            ic_arr = np.array(ic_values, dtype=float)
            rolling_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr)) if len(ic_arr) > 1 else 0.0

            # Trend detection via linear regression
            trend_label = "unknown"
            trend_slope = 0.0
            if len(ic_arr) >= 3:
                x = np.arange(len(ic_arr), dtype=float)
                coeffs = np.polyfit(x, ic_arr, 1)
                trend_slope = float(coeffs[0])
                if trend_slope > 0.001:
                    trend_label = "improving"
                elif trend_slope < -0.001:
                    trend_label = "degrading"
                else:
                    trend_label = "stable"

            raw = {
                "rolling_ic_mean": round(rolling_mean, 6),
                "ic_std": round(ic_std, 6),
                "n_cycles": len(ic_values),
                "trend": trend_label,
                "trend_slope": round(trend_slope, 6),
                "latest_ic": round(float(ic_arr[-1]), 6),
                "ic_values": [round(float(v), 6) for v in ic_arr],
            }

            # Scoring
            if rolling_mean >= 0.03:
                return HealthCheckResult(
                    name=name, domain=domain, score=90.0, status="PASS",
                    explanation=(
                        f"Strong IC: mean={rolling_mean:.4f} over {len(ic_values)} cycles "
                        f"(trend: {trend_label})"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, raw_value=rolling_mean,
                )
            if rolling_mean >= IC_TRACKING_WARN_THRESHOLD:
                return HealthCheckResult(
                    name=name, domain=domain, score=70.0, status="PASS",
                    explanation=(
                        f"Acceptable IC: mean={rolling_mean:.4f} over {len(ic_values)} cycles "
                        f"(trend: {trend_label})"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, raw_value=rolling_mean,
                )
            if rolling_mean >= IC_TRACKING_CRITICAL_THRESHOLD:
                return HealthCheckResult(
                    name=name, domain=domain, score=40.0, status="WARN",
                    explanation=(
                        f"Weak IC: mean={rolling_mean:.4f} < {IC_TRACKING_WARN_THRESHOLD} "
                        f"over {len(ic_values)} cycles (trend: {trend_label})"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, raw_value=rolling_mean,
                )
            return HealthCheckResult(
                name=name, domain=domain, score=15.0, status="FAIL",
                explanation=(
                    f"CRITICAL: Negative IC mean={rolling_mean:.4f} — model is anti-predictive "
                    f"over {len(ic_values)} cycles (trend: {trend_label})"
                ),
                methodology=methodology, raw_metrics=raw,
                thresholds=thresholds, severity="critical",
                raw_value=rolling_mean,
            )

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")

    @staticmethod
    def _extract_ic_from_cycle_report() -> Optional[float]:
        """Extract best IC mean from the latest autopilot cycle report.

        Scans all top_decisions in latest_cycle.json for ic_mean values
        and returns the best (highest) one.  Returns None if no IC data
        found.
        """
        try:
            from quant_engine.config import AUTOPILOT_CYCLE_REPORT
            report_path = Path(AUTOPILOT_CYCLE_REPORT)
            if not report_path.exists():
                return None

            with open(report_path, "r") as f:
                report = json.load(f)

            ic_values = []
            for decision in report.get("top_decisions", []):
                metrics = decision.get("metrics", {})
                ic = metrics.get("ic_mean")
                if ic is not None:
                    ic_values.append(float(ic))

            return max(ic_values) if ic_values else None
        except Exception:
            return None

    # ── Ensemble Disagreement Tracking (SPEC-H02) ─────────────────────

    _DISAGREEMENT_DB_PATH: Optional[Path] = None
    _MAX_DISAGREEMENT_SNAPSHOTS = 200  # Retain up to 200 prediction cycle records

    @classmethod
    def _get_disagreement_db_path(cls) -> Path:
        """Return the path to the disagreement tracking SQLite database."""
        if cls._DISAGREEMENT_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._DISAGREEMENT_DB_PATH = Path(RESULTS_DIR) / "disagreement_tracking.db"
        return cls._DISAGREEMENT_DB_PATH

    @classmethod
    def _ensure_disagreement_table(cls) -> None:
        """Create disagreement_history table if it doesn't exist."""
        import sqlite3
        db_path = cls._get_disagreement_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS disagreement_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mean_disagreement REAL NOT NULL,
                    max_disagreement REAL,
                    n_members INTEGER,
                    n_assets INTEGER,
                    pct_high_disagreement REAL,
                    member_names TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_disagreement_snapshot(
        self,
        mean_disagreement: float,
        max_disagreement: Optional[float] = None,
        n_members: int = 0,
        n_assets: int = 0,
        pct_high_disagreement: float = 0.0,
        member_names: Optional[List[str]] = None,
    ) -> None:
        """Save ensemble disagreement metrics from a prediction cycle.

        Called after EnsemblePredictor.predict() to track disagreement
        across ensemble members over time.

        Parameters
        ----------
        mean_disagreement : float
            Mean std of predictions across ensemble members (per-row, then averaged).
        max_disagreement : float, optional
            Maximum per-row disagreement observed.
        n_members : int
            Number of ensemble members that produced predictions.
        n_assets : int
            Number of assets predicted in this cycle.
        pct_high_disagreement : float
            Fraction of rows where disagreement exceeds the warning threshold.
        member_names : list of str, optional
            Names of ensemble members (e.g. ["global", "regime_0", "regime_1"]).
        """
        import sqlite3
        try:
            self._ensure_disagreement_table()
            db_path = self._get_disagreement_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                timestamp = datetime.now(timezone.utc).isoformat()
                names_str = json.dumps(member_names) if member_names else "[]"
                conn.execute(
                    "INSERT INTO disagreement_history "
                    "(timestamp, mean_disagreement, max_disagreement, n_members, "
                    "n_assets, pct_high_disagreement, member_names) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, mean_disagreement, max_disagreement,
                     n_members, n_assets, pct_high_disagreement, names_str),
                )
                # Prune old entries
                conn.execute(
                    "DELETE FROM disagreement_history WHERE id NOT IN "
                    "(SELECT id FROM disagreement_history ORDER BY id DESC LIMIT ?)",
                    (self._MAX_DISAGREEMENT_SNAPSHOTS,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to save disagreement snapshot: %s", e)

    def get_disagreement_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent disagreement snapshots for tracking and health checks.

        Parameters
        ----------
        limit : int
            Maximum number of recent snapshots to return.

        Returns
        -------
        List of dicts, each with: timestamp, mean_disagreement,
        max_disagreement, n_members, n_assets, pct_high_disagreement,
        member_names.  Ordered chronologically (oldest first).
        """
        import sqlite3
        try:
            self._ensure_disagreement_table()
            db_path = self._get_disagreement_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.execute(
                    "SELECT timestamp, mean_disagreement, max_disagreement, "
                    "n_members, n_assets, pct_high_disagreement, member_names "
                    "FROM disagreement_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                history = []
                for (ts, mean_d, max_d, n_mem, n_a,
                     pct_high, names_str) in reversed(rows):
                    try:
                        member_names_list = json.loads(names_str) if names_str else []
                    except (json.JSONDecodeError, TypeError):
                        member_names_list = []
                    history.append({
                        "timestamp": ts,
                        "mean_disagreement": mean_d,
                        "max_disagreement": max_d,
                        "n_members": n_mem,
                        "n_assets": n_a,
                        "pct_high_disagreement": pct_high,
                        "member_names": member_names_list,
                    })
                return history
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to load disagreement history: %s", e)
            return []

    # ── Execution Quality Monitoring (SPEC-H03) ───────────────────────

    _EXEC_QUALITY_DB_PATH: Optional[Path] = None
    _MAX_EXEC_QUALITY_RECORDS = 500  # Retain up to 500 fill quality records

    @classmethod
    def _get_exec_quality_db_path(cls) -> Path:
        """Return the path to the execution quality SQLite database."""
        if cls._EXEC_QUALITY_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._EXEC_QUALITY_DB_PATH = Path(RESULTS_DIR) / "exec_quality.db"
        return cls._EXEC_QUALITY_DB_PATH

    @classmethod
    def _ensure_exec_quality_table(cls) -> None:
        """Create exec_quality table if it doesn't exist."""
        import sqlite3
        db_path = cls._get_exec_quality_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exec_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    predicted_cost_bps REAL NOT NULL,
                    actual_cost_bps REAL NOT NULL,
                    cost_surprise_bps REAL NOT NULL,
                    fill_ratio REAL,
                    participation_rate REAL,
                    regime INTEGER
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_execution_quality_fill(
        self,
        symbol: str,
        side: str,
        predicted_cost_bps: float,
        actual_cost_bps: float,
        fill_ratio: Optional[float] = None,
        participation_rate: Optional[float] = None,
        regime: Optional[int] = None,
    ) -> None:
        """Save a paper-trade fill's execution quality to the tracking database.

        Called by the paper trader after each fill to accumulate
        predicted-vs-actual cost data for health monitoring.

        Parameters
        ----------
        symbol : str
            Ticker or PERMNO identifier.
        side : str
            "buy" or "sell".
        predicted_cost_bps : float
            Model-predicted total execution cost in basis points.
        actual_cost_bps : float
            Realized execution cost in basis points.
        fill_ratio : float, optional
            Fraction of desired notional filled (0-1).
        participation_rate : float, optional
            Fraction of daily volume executed.
        regime : int, optional
            Market regime code at time of fill.
        """
        import sqlite3
        try:
            self._ensure_exec_quality_table()
            db_path = self._get_exec_quality_db_path()
            cost_surprise = float(actual_cost_bps) - float(predicted_cost_bps)
            conn = sqlite3.connect(str(db_path))
            try:
                timestamp = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO exec_quality "
                    "(timestamp, symbol, side, predicted_cost_bps, actual_cost_bps, "
                    "cost_surprise_bps, fill_ratio, participation_rate, regime) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        str(symbol),
                        str(side),
                        float(predicted_cost_bps),
                        float(actual_cost_bps),
                        cost_surprise,
                        float(fill_ratio) if fill_ratio is not None else None,
                        float(participation_rate) if participation_rate is not None else None,
                        int(regime) if regime is not None else None,
                    ),
                )
                # Prune old entries
                conn.execute(
                    "DELETE FROM exec_quality WHERE id NOT IN "
                    "(SELECT id FROM exec_quality ORDER BY id DESC LIMIT ?)",
                    (self._MAX_EXEC_QUALITY_RECORDS,),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to save execution quality fill: %s", e)

    def get_execution_quality_history(
        self, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent execution quality fill records.

        Parameters
        ----------
        limit : int
            Maximum number of recent records to return.

        Returns
        -------
        List of dicts, each with: timestamp, symbol, side,
        predicted_cost_bps, actual_cost_bps, cost_surprise_bps,
        fill_ratio, participation_rate, regime.
        Ordered chronologically (oldest first).
        """
        import sqlite3
        try:
            self._ensure_exec_quality_table()
            db_path = self._get_exec_quality_db_path()
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.execute(
                    "SELECT timestamp, symbol, side, predicted_cost_bps, "
                    "actual_cost_bps, cost_surprise_bps, fill_ratio, "
                    "participation_rate, regime "
                    "FROM exec_quality ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                history = []
                for (ts, sym, side, pred, actual, surprise,
                     fill_r, part_r, reg) in reversed(rows):
                    history.append({
                        "timestamp": ts,
                        "symbol": sym,
                        "side": side,
                        "predicted_cost_bps": pred,
                        "actual_cost_bps": actual,
                        "cost_surprise_bps": surprise,
                        "fill_ratio": fill_r,
                        "participation_rate": part_r,
                        "regime": reg,
                    })
                return history
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to load execution quality history: %s", e)
            return []

    def _check_cost_surprise(self) -> HealthCheckResult:
        """Track rolling cost surprise across paper-trade fills (SPEC-H03).

        Retrieves last N paper-trade fills, computes cost surprise
        (actual_cost_bps - predicted_cost_bps), and flags:
            - WARNING if mean cost surprise > 2 bps (model underestimates)
            - CRITICAL (FAIL) if mean cost surprise > 5 bps

        Negative cost surprise means the model is conservative (overestimates
        costs), which is acceptable behavior.

        Score mapping:
            surprise <= 0     → 90 (model conservative or accurate)
            surprise <= warn  → 70 (slight underestimation, acceptable)
            surprise <= crit  → 40 (meaningful underestimation — WARNING)
            surprise >  crit  → 15 (severe underestimation — CRITICAL)
        """
        domain = "execution_quality"
        name = "cost_surprise"
        methodology = (
            "Retrieve recent paper-trade fills and compute cost surprise "
            "(actual_cost_bps - predicted_cost_bps) for each fill.  "
            "Positive surprise means the execution cost model underestimated "
            "costs.  Track rolling mean and linear trend.  Alert if "
            "systematically positive, indicating the model is too optimistic "
            "about execution costs."
        )
        try:
            from quant_engine.config import (
                EXEC_QUALITY_LOOKBACK,
                EXEC_QUALITY_WARN_SURPRISE_BPS,
                EXEC_QUALITY_CRITICAL_SURPRISE_BPS,
            )
        except ImportError:
            EXEC_QUALITY_LOOKBACK = 50
            EXEC_QUALITY_WARN_SURPRISE_BPS = 2.0
            EXEC_QUALITY_CRITICAL_SURPRISE_BPS = 5.0

        thresholds = {
            "warn_surprise_bps": EXEC_QUALITY_WARN_SURPRISE_BPS,
            "critical_surprise_bps": EXEC_QUALITY_CRITICAL_SURPRISE_BPS,
        }

        try:
            history = self.get_execution_quality_history(
                limit=EXEC_QUALITY_LOOKBACK,
            )
            surprises = [
                h["cost_surprise_bps"]
                for h in history
                if h["cost_surprise_bps"] is not None
            ]

            if not surprises:
                return _unavailable(
                    name, domain,
                    "No execution quality data — run paper trades to collect "
                    "predicted-vs-actual fill costs",
                )

            arr = np.array(surprises, dtype=float)
            mean_surprise = float(np.mean(arr))
            median_surprise = float(np.median(arr))
            std_surprise = float(np.std(arr)) if len(arr) > 1 else 0.0
            pct_underestimated = float(np.mean(arr > 0))

            # Trend detection via linear regression
            trend_label = "unknown"
            trend_slope = 0.0
            if len(arr) >= 3:
                x = np.arange(len(arr), dtype=float)
                coeffs = np.polyfit(x, arr, 1)
                trend_slope = float(coeffs[0])
                if trend_slope > 0.05:
                    trend_label = "worsening"
                elif trend_slope < -0.05:
                    trend_label = "improving"
                else:
                    trend_label = "stable"

            raw = {
                "mean_surprise_bps": round(mean_surprise, 3),
                "median_surprise_bps": round(median_surprise, 3),
                "std_surprise_bps": round(std_surprise, 3),
                "pct_underestimated": round(pct_underestimated, 3),
                "n_fills": len(surprises),
                "trend": trend_label,
                "trend_slope_bps_per_fill": round(trend_slope, 4),
                "latest_surprise_bps": round(float(arr[-1]), 3),
            }

            # Scoring
            if mean_surprise <= 0:
                return HealthCheckResult(
                    name=name, domain=domain, score=90.0, status="PASS",
                    explanation=(
                        f"Cost model conservative: mean surprise="
                        f"{mean_surprise:+.2f} bps over {len(surprises)} fills "
                        f"(trend: {trend_label})"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, raw_value=mean_surprise,
                )
            if mean_surprise <= EXEC_QUALITY_WARN_SURPRISE_BPS:
                return HealthCheckResult(
                    name=name, domain=domain, score=70.0, status="PASS",
                    explanation=(
                        f"Slight cost underestimation: mean surprise="
                        f"{mean_surprise:+.2f} bps over {len(surprises)} fills "
                        f"(trend: {trend_label})"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, raw_value=mean_surprise,
                )
            if mean_surprise <= EXEC_QUALITY_CRITICAL_SURPRISE_BPS:
                return HealthCheckResult(
                    name=name, domain=domain, score=40.0, status="WARN",
                    explanation=(
                        f"Model underestimates costs: mean surprise="
                        f"{mean_surprise:+.2f} bps > {EXEC_QUALITY_WARN_SURPRISE_BPS} "
                        f"over {len(surprises)} fills (trend: {trend_label})"
                    ),
                    methodology=methodology, raw_metrics=raw,
                    thresholds=thresholds, raw_value=mean_surprise,
                )
            return HealthCheckResult(
                name=name, domain=domain, score=15.0, status="FAIL",
                explanation=(
                    f"CRITICAL: Severe cost underestimation: mean surprise="
                    f"{mean_surprise:+.2f} bps > {EXEC_QUALITY_CRITICAL_SURPRISE_BPS} "
                    f"over {len(surprises)} fills (trend: {trend_label})"
                ),
                methodology=methodology, raw_metrics=raw,
                thresholds=thresholds, severity="critical",
                raw_value=mean_surprise,
            )

        except Exception as e:
            logger.warning("Health check '%s' failed: %s", name, e)
            return _unavailable(name, domain, f"Error: {e}")
