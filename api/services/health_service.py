"""System health assessment for API consumption.

Thin orchestrator that coordinates health checks, scoring, storage, and alerts
across 5 weighted domains. Implementation details are in decomposed modules:
    - health_checks.py  — standalone check functions
    - health_scoring.py — domain score aggregation
    - health_storage.py — SQLite persistence and trend computation
    - health_alerts.py  — alert evaluation and dispatch

SPEC_AUDIT_FIX_04 changes:
    T1: Snapshot persistence throttled to once per 5 minutes
    T2: Heavy cache scans run in background, cached results served inline
    T3: Trend detection thresholds configurable via api/config.py
    T4: Feature stability path uses RESULTS_DIR from config
    T5: HealthService decomposed into focused modules
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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


HEALTH_SNAPSHOT_MIN_INTERVAL_SECONDS: int = 300  # 5 minutes


class HealthService:
    """Computes system health from model age, cache freshness, and runtime metrics."""

    def __init__(self) -> None:
        self._last_snapshot_time: float = 0.0
        # Cached background scan results (T2)
        self._cached_anomaly_result: Optional[HealthCheckResult] = None
        self._cached_microstructure_result: Optional[HealthCheckResult] = None
        self._cache_scan_timestamp: float = 0.0

    # ── Background Scan Helpers (T2) ──────────────────────────────────

    def _get_cached_anomaly_result(self) -> HealthCheckResult:
        """Return cached anomaly scan result, or pending placeholder."""
        if self._cached_anomaly_result is not None:
            return self._cached_anomaly_result
        return HealthCheckResult(
            name="data_anomalies", domain="data_integrity", score=0.0,
            status="UNAVAILABLE", explanation="Initial scan in progress",
            methodology="N/A — background scan pending", data_available=False,
            raw_metrics={"status": "pending", "message": "Initial scan in progress"},
        )

    def _get_cached_microstructure_result(self) -> HealthCheckResult:
        """Return cached microstructure scan result, or pending placeholder."""
        if self._cached_microstructure_result is not None:
            return self._cached_microstructure_result
        return HealthCheckResult(
            name="microstructure", domain="data_integrity", score=0.0,
            status="UNAVAILABLE", explanation="Initial scan in progress",
            methodology="N/A — background scan pending", data_available=False,
            raw_metrics={"status": "pending", "message": "Initial scan in progress"},
        )

    def run_background_scans(self) -> None:
        """Execute heavy cache scans and cache the results."""
        from . import health_checks as hc

        try:
            self._cached_anomaly_result = hc.check_data_quality_anomalies()
        except Exception as e:
            logger.warning("Background anomaly scan failed: %s", e)
        try:
            self._cached_microstructure_result = hc.check_market_microstructure()
        except Exception as e:
            logger.warning("Background microstructure scan failed: %s", e)
        self._cache_scan_timestamp = _time.time()
        logger.info("Background health scans completed at %s",
                     datetime.now(timezone.utc).isoformat())

    # ── Quick Status ──────────────────────────────────────────────────

    def get_quick_status(self) -> Dict[str, Any]:
        """Lightweight health check for ``GET /api/health``."""
        from quant_engine.config import DATA_CACHE_DIR, MODEL_DIR

        status = "healthy"
        checks: Dict[str, str] = {}

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

        model_dir = Path(MODEL_DIR)
        registry_path = model_dir / "registry.json"
        if registry_path.exists():
            checks["model_registry"] = "present"
        else:
            checks["model_registry"] = "missing"
            if status == "healthy":
                status = "degraded"

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

        return {"status": status, "checks": checks,
                "timestamp": datetime.now(timezone.utc).isoformat()}

    # ── Detailed Health ───────────────────────────────────────────────

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

        runtime_health = self.compute_comprehensive_health()
        result["runtime_health"] = runtime_health

        # Save snapshot for history tracking (throttled — T1)
        now = _time.time()
        if now - self._last_snapshot_time >= HEALTH_SNAPSHOT_MIN_INTERVAL_SECONDS:
            self.save_health_snapshot(runtime_health)
            self._last_snapshot_time = now

        result["health_history"] = self.get_health_history()
        return result

    # ── Domain Scoring (delegates to health_scoring) ──────────────────

    @staticmethod
    def _domain_score(checks: List[HealthCheckResult]) -> Optional[float]:
        from .health_scoring import domain_score
        return domain_score(checks)

    @staticmethod
    def _domain_status(checks: List[HealthCheckResult]) -> str:
        from .health_scoring import domain_status
        return domain_status(checks)

    # ── Comprehensive Health ─────────────────────────────────────────

    def compute_comprehensive_health(self) -> Dict[str, Any]:
        """Compute comprehensive health score across 5 weighted domains."""
        from .health_confidence import HealthConfidenceCalculator
        from . import health_checks as hc

        ci_calc = HealthConfidenceCalculator()
        domains: Dict[str, Dict[str, Any]] = {}

        # ── Data Integrity (25%) ──
        data_checks = [
            hc.check_survivorship_bias(),
            self._get_cached_anomaly_result(),
            self._get_cached_microstructure_result(),
            hc.check_wrds_status(),
        ]
        for c in data_checks:
            c.severity = "critical"
        domains["data_integrity"] = {
            "weight": 0.25,
            "score": self._domain_score(data_checks),
            "status": self._domain_status(data_checks),
            "checks": [c.to_dict() for c in data_checks],
        }

        # ── Signal Quality (25%) ──
        signal_checks = [
            hc.check_signal_decay(),
            hc.check_prediction_distribution(),
            hc.check_ensemble_disagreement(self.get_disagreement_history),
            hc.check_information_ratio(),
            hc.check_ic_tracking(self.get_ic_history),
        ]
        for c in signal_checks:
            c.severity = "critical"
        domains["signal_quality"] = {
            "weight": 0.25,
            "score": self._domain_score(signal_checks),
            "status": self._domain_status(signal_checks),
            "checks": [c.to_dict() for c in signal_checks],
        }

        # ── Risk Management (20%) ──
        risk_checks = [
            hc.check_tail_risk(),
            hc.check_correlation_regime(),
            hc.check_capital_utilization(),
        ]
        for c in risk_checks:
            c.severity = "standard"
        domains["risk_management"] = {
            "weight": 0.20,
            "score": self._domain_score(risk_checks),
            "status": self._domain_status(risk_checks),
            "checks": [c.to_dict() for c in risk_checks],
        }

        # ── Execution Quality (20%) ──
        exec_checks = [
            hc.check_execution_quality(),
            hc.check_signal_profitability(),
            hc.check_cost_surprise(self.get_execution_quality_history),
        ]
        for c in exec_checks:
            c.severity = "standard"
        domains["execution_quality"] = {
            "weight": 0.20,
            "score": self._domain_score(exec_checks),
            "status": self._domain_status(exec_checks),
            "checks": [c.to_dict() for c in exec_checks],
        }

        # ── Model Governance (10%) ──
        gov_checks = [
            hc.check_feature_importance_drift(),
            hc.check_cv_gap_trend(),
            hc.check_regime_transition_health(),
            hc.check_retraining_effectiveness(),
        ]
        for c in gov_checks:
            c.severity = "informational"
        domains["model_governance"] = {
            "weight": 0.10,
            "score": self._domain_score(gov_checks),
            "status": self._domain_status(gov_checks),
            "checks": [c.to_dict() for c in gov_checks],
        }

        # Count available/total
        all_checks = data_checks + signal_checks + risk_checks + exec_checks + gov_checks
        checks_total = len(all_checks)
        checks_available = sum(1 for c in all_checks if c.status != "UNAVAILABLE")

        # Weighted overall
        from .health_scoring import compute_overall_score
        overall, overall_status = compute_overall_score(domains)

        # ── Confidence intervals (Spec 09) ──
        domain_ci: Dict[str, Dict[str, Any]] = {}
        for dname, dinfo in domains.items():
            avail = [c for c in dinfo["checks"] if c.get("status") != "UNAVAILABLE"]
            scores = [c.get("score", 0.0) for c in avail]
            if scores:
                ci_result = ci_calc.compute_ci(samples=np.array(scores))
                domain_ci[dname] = ci_result.to_dict()
            else:
                domain_ci[dname] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                                    "n_samples": 0, "method": "insufficient",
                                    "low_confidence": True, "ci_width": 0.0}

        overall_ci = None
        available_domains = {k: v for k, v in domains.items() if v["score"] is not None}
        if overall is not None and available_domains:
            d_scores, d_widths, d_weights = [], [], []
            for dname, dinfo in available_domains.items():
                d_scores.append(dinfo["score"])
                ci = domain_ci.get(dname, {})
                d_widths.append(ci.get("ci_width", 0.0))
                d_weights.append(dinfo["weight"])
            from .health_confidence import HealthConfidenceCalculator as HCC
            ci_lower, ci_upper = HCC.propagate_weighted_ci(d_scores, d_widths, d_weights)
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

    # ── Alert Integration ─────────────────────────────────────────────

    def _run_alerts(self, overall_score: Optional[float],
                    domains: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            from .health_alerts import create_alert_manager
            alert_mgr = create_alert_manager()
            pending_alerts = []
            domain_scores = {d: v.get("score") for d, v in domains.items()
                            if v.get("score") is not None}
            pending_alerts.extend(alert_mgr.check_domain_failures(domain_scores))
            if overall_score is not None:
                history = self.get_health_history(limit=2)
                if history:
                    yesterday_score = history[-1].get("overall_score")
                    if yesterday_score is not None:
                        deg = alert_mgr.check_health_degradation(overall_score, yesterday_score)
                        if deg is not None:
                            pending_alerts.append(deg)
            return [a.to_dict() for a in alert_mgr.process_alerts(pending_alerts)]
        except Exception as e:
            logger.warning("Alert processing failed: %s", e)
            return []

    # ── Storage delegates (backward-compatible API) ───────────────────

    _HEALTH_DB_PATH: Optional[Path] = None
    _IC_DB_PATH: Optional[Path] = None
    _DISAGREEMENT_DB_PATH: Optional[Path] = None
    _EXEC_QUALITY_DB_PATH: Optional[Path] = None
    _MAX_SNAPSHOTS = 90
    _MAX_IC_SNAPSHOTS = 200
    _MAX_DISAGREEMENT_SNAPSHOTS = 200
    _MAX_EXEC_QUALITY_RECORDS = 500

    @classmethod
    def _get_health_db_path(cls) -> Path:
        if cls._HEALTH_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._HEALTH_DB_PATH = Path(RESULTS_DIR) / "health_history.db"
        return cls._HEALTH_DB_PATH

    @classmethod
    def _ensure_health_table(cls) -> None:
        from .health_storage import _ensure_health_table
        _ensure_health_table(cls._get_health_db_path())

    def save_health_snapshot(self, health_result: Dict[str, Any]) -> None:
        from .health_storage import save_health_snapshot
        save_health_snapshot(health_result)

    def get_health_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        from .health_storage import get_health_history
        return get_health_history(limit=limit)

    def get_health_history_with_trends(self, limit: int = 90) -> Dict[str, Any]:
        from .health_storage import get_health_history_with_trends
        return get_health_history_with_trends(limit=limit)

    @staticmethod
    def _compute_rolling_average(scores: List[float], window: int = 7) -> List[float]:
        from .health_storage import compute_rolling_average
        return compute_rolling_average(scores, window=window)

    @staticmethod
    def _detect_trend(scores: List[float], window: int = 30) -> tuple:
        from .health_storage import detect_trend
        return detect_trend(scores, window=window)

    # IC tracking
    @classmethod
    def _get_ic_db_path(cls) -> Path:
        if cls._IC_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._IC_DB_PATH = Path(RESULTS_DIR) / "ic_tracking.db"
        return cls._IC_DB_PATH

    @classmethod
    def _ensure_ic_table(cls) -> None:
        from .health_storage import _ensure_ic_table
        _ensure_ic_table(cls._get_ic_db_path())

    def save_ic_snapshot(self, ic_mean: float, ic_ir: Optional[float] = None,
                         n_candidates: int = 0, n_passed: int = 0,
                         best_strategy_id: str = "") -> None:
        from .health_storage import save_ic_snapshot
        save_ic_snapshot(ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id)

    def get_ic_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        from .health_storage import get_ic_history
        return get_ic_history(limit=limit)

    def _check_ic_tracking(self) -> HealthCheckResult:
        from .health_checks import check_ic_tracking
        return check_ic_tracking(self.get_ic_history)

    @staticmethod
    def _extract_ic_from_cycle_report() -> Optional[float]:
        from .health_checks import _extract_ic_from_cycle_report
        return _extract_ic_from_cycle_report()

    # Disagreement tracking
    @classmethod
    def _get_disagreement_db_path(cls) -> Path:
        if cls._DISAGREEMENT_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._DISAGREEMENT_DB_PATH = Path(RESULTS_DIR) / "disagreement_tracking.db"
        return cls._DISAGREEMENT_DB_PATH

    @classmethod
    def _ensure_disagreement_table(cls) -> None:
        from .health_storage import _ensure_disagreement_table
        _ensure_disagreement_table(cls._get_disagreement_db_path())

    def save_disagreement_snapshot(self, mean_disagreement: float,
                                   max_disagreement: Optional[float] = None,
                                   n_members: int = 0, n_assets: int = 0,
                                   pct_high_disagreement: float = 0.0,
                                   member_names: Optional[List[str]] = None) -> None:
        from .health_storage import save_disagreement_snapshot
        save_disagreement_snapshot(mean_disagreement, max_disagreement,
                                   n_members, n_assets, pct_high_disagreement,
                                   member_names)

    def get_disagreement_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        from .health_storage import get_disagreement_history
        return get_disagreement_history(limit=limit)

    def _check_ensemble_disagreement(self) -> HealthCheckResult:
        from .health_checks import check_ensemble_disagreement
        return check_ensemble_disagreement(self.get_disagreement_history)

    def _check_ensemble_disagreement_static_fallback(self, name, domain,
                                                      methodology, thresholds):
        from .health_checks import _check_ensemble_disagreement_static_fallback
        return _check_ensemble_disagreement_static_fallback(name, domain, methodology, thresholds)

    # Execution quality tracking
    @classmethod
    def _get_exec_quality_db_path(cls) -> Path:
        if cls._EXEC_QUALITY_DB_PATH is None:
            from quant_engine.config import RESULTS_DIR
            cls._EXEC_QUALITY_DB_PATH = Path(RESULTS_DIR) / "exec_quality.db"
        return cls._EXEC_QUALITY_DB_PATH

    @classmethod
    def _ensure_exec_quality_table(cls) -> None:
        from .health_storage import _ensure_exec_quality_table
        _ensure_exec_quality_table(cls._get_exec_quality_db_path())

    def save_execution_quality_fill(self, symbol: str, side: str,
                                     predicted_cost_bps: float,
                                     actual_cost_bps: float,
                                     fill_ratio: Optional[float] = None,
                                     participation_rate: Optional[float] = None,
                                     regime: Optional[int] = None) -> None:
        from .health_storage import save_execution_quality_fill
        save_execution_quality_fill(symbol, side, predicted_cost_bps, actual_cost_bps,
                                     fill_ratio, participation_rate, regime)

    def get_execution_quality_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        from .health_storage import get_execution_quality_history
        return get_execution_quality_history(limit=limit)

    def _check_cost_surprise(self) -> HealthCheckResult:
        from .health_checks import check_cost_surprise
        return check_cost_surprise(self.get_execution_quality_history)

    # Remaining check delegates
    def _check_survivorship_bias(self) -> HealthCheckResult:
        from .health_checks import check_survivorship_bias
        return check_survivorship_bias()

    def _check_data_quality_anomalies(self) -> HealthCheckResult:
        from .health_checks import check_data_quality_anomalies
        return check_data_quality_anomalies()

    def _check_market_microstructure(self) -> HealthCheckResult:
        from .health_checks import check_market_microstructure
        return check_market_microstructure()

    def _check_wrds_status(self) -> HealthCheckResult:
        from .health_checks import check_wrds_status
        return check_wrds_status()

    def _check_signal_decay(self) -> HealthCheckResult:
        from .health_checks import check_signal_decay
        return check_signal_decay()

    def _check_prediction_distribution(self) -> HealthCheckResult:
        from .health_checks import check_prediction_distribution
        return check_prediction_distribution()

    def _check_information_ratio(self) -> HealthCheckResult:
        from .health_checks import check_information_ratio
        return check_information_ratio()

    def _check_tail_risk(self) -> HealthCheckResult:
        from .health_checks import check_tail_risk
        return check_tail_risk()

    def _check_correlation_regime(self) -> HealthCheckResult:
        from .health_checks import check_correlation_regime
        return check_correlation_regime()

    def _check_capital_utilization(self) -> HealthCheckResult:
        from .health_checks import check_capital_utilization
        return check_capital_utilization()

    def _check_execution_quality(self) -> HealthCheckResult:
        from .health_checks import check_execution_quality
        return check_execution_quality()

    def _check_signal_profitability(self) -> HealthCheckResult:
        from .health_checks import check_signal_profitability
        return check_signal_profitability()

    def _check_cv_gap_trend(self) -> HealthCheckResult:
        from .health_checks import check_cv_gap_trend
        return check_cv_gap_trend()

    def _check_retraining_effectiveness(self) -> HealthCheckResult:
        from .health_checks import check_retraining_effectiveness
        return check_retraining_effectiveness()

    def _check_feature_importance_drift(self) -> HealthCheckResult:
        from .health_checks import check_feature_importance_drift
        return check_feature_importance_drift()

    def _check_regime_transition_health(self) -> HealthCheckResult:
        from .health_checks import check_regime_transition_health
        return check_regime_transition_health()

    @staticmethod
    def _load_trades_csv():
        from .health_checks import load_trades_csv
        return load_trades_csv()

    @staticmethod
    def _load_benchmark_returns():
        from .health_checks import load_benchmark_returns
        return load_benchmark_returns()
