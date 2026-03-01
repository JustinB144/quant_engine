"""System health assessment — thin orchestrator (SPEC_AUDIT_FIX_04 T5)."""
from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .health_types import HealthCheckResult, _unavailable, _SEVERITY_WEIGHTS  # noqa: F401

logger = logging.getLogger(__name__)
HEALTH_SNAPSHOT_MIN_INTERVAL_SECONDS: int = 300  # T1: 5-minute cooldown
_HISTORY_ARG_CHECKS = {"check_ensemble_disagreement": "get_disagreement_history",
                       "check_ic_tracking": "get_ic_history",
                       "check_cost_surprise": "get_execution_quality_history"}
_DB_FILES = {"_HEALTH_DB_PATH": "health_history.db", "_IC_DB_PATH": "ic_tracking.db",
             "_DISAGREEMENT_DB_PATH": "disagreement_tracking.db",
             "_EXEC_QUALITY_DB_PATH": "exec_quality.db"}


def _build_domain(checks: List[HealthCheckResult], weight: float,
                  severity: str) -> Dict[str, Any]:
    for c in checks:
        c.severity = severity
    from .health_scoring import domain_score, domain_status
    return {"weight": weight, "score": domain_score(checks),
            "status": domain_status(checks), "checks": [c.to_dict() for c in checks]}


class HealthService:
    """Computes system health from model age, cache freshness, and runtime metrics."""
    _HEALTH_DB_PATH: Optional[Path] = None
    _IC_DB_PATH: Optional[Path] = None
    _DISAGREEMENT_DB_PATH: Optional[Path] = None
    _EXEC_QUALITY_DB_PATH: Optional[Path] = None
    _MAX_SNAPSHOTS = 90
    _MAX_IC_SNAPSHOTS = 200
    _MAX_DISAGREEMENT_SNAPSHOTS = 200
    _MAX_EXEC_QUALITY_RECORDS = 500

    def __init__(self) -> None:
        self._last_snapshot_time: float = 0.0
        self._cached_anomaly_result: Optional[HealthCheckResult] = None
        self._cached_microstructure_result: Optional[HealthCheckResult] = None
        self._cache_scan_timestamp: float = 0.0

    def __getattr__(self, name: str):
        if name.startswith("_check_"):
            from . import health_checks as hc
            fn_name = name[1:]  # _check_X -> check_X
            fn = getattr(hc, fn_name, None)
            if fn is not None:
                hist_attr = _HISTORY_ARG_CHECKS.get(fn_name)
                if hist_attr:
                    return lambda: fn(getattr(self, hist_attr))
                return fn
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def _get_cached_anomaly_result(self) -> HealthCheckResult:
        if self._cached_anomaly_result is not None:
            return self._cached_anomaly_result
        return _unavailable("data_anomalies", "data_integrity", "Initial scan in progress")

    def _get_cached_microstructure_result(self) -> HealthCheckResult:
        if self._cached_microstructure_result is not None:
            return self._cached_microstructure_result
        return _unavailable("microstructure", "data_integrity", "Initial scan in progress")

    def run_background_scans(self) -> None:
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

    def get_quick_status(self) -> Dict[str, Any]:
        from .health_checks import quick_status
        return quick_status()

    def get_detailed_health(self) -> Dict[str, Any]:
        from .data_helpers import collect_health_data
        payload = collect_health_data()
        result: Dict[str, Any] = {
            "overall_score": payload.overall_score, "overall_status": payload.overall_status,
            "generated_at": payload.generated_at.isoformat(),
            "data_integrity_score": payload.data_integrity_score,
            "promotion_score": payload.promotion_score, "wf_score": payload.wf_score,
            "execution_score": payload.execution_score,
            "complexity_score": payload.complexity_score,
        }
        for section in ["survivorship_checks", "data_quality_checks", "promotion_checks",
                        "wf_checks", "execution_checks", "complexity_checks", "strengths"]:
            result[section] = [{"name": c.name, "status": c.status, "detail": c.detail,
                                "value": c.value, "recommendation": c.recommendation}
                               for c in getattr(payload, section, [])]
        result["promotion_funnel"] = payload.promotion_funnel
        result["feature_inventory"] = payload.feature_inventory
        result["knob_inventory"] = payload.knob_inventory
        runtime_health = self.compute_comprehensive_health()
        result["runtime_health"] = runtime_health
        now = _time.time()
        if now - self._last_snapshot_time >= HEALTH_SNAPSHOT_MIN_INTERVAL_SECONDS:
            self.save_health_snapshot(runtime_health)
            self._last_snapshot_time = now
        result["health_history"] = self.get_health_history()
        return result

    def compute_comprehensive_health(self) -> Dict[str, Any]:
        from . import health_checks as hc
        from .health_scoring import compute_overall_score, compute_confidence_intervals
        domains: Dict[str, Dict[str, Any]] = {}
        domains["data_integrity"] = _build_domain([
            hc.check_survivorship_bias(), self._get_cached_anomaly_result(),
            self._get_cached_microstructure_result(), hc.check_wrds_status(),
        ], 0.25, "critical")
        domains["signal_quality"] = _build_domain([
            hc.check_signal_decay(), hc.check_prediction_distribution(),
            hc.check_ensemble_disagreement(self.get_disagreement_history),
            hc.check_information_ratio(), hc.check_ic_tracking(self.get_ic_history),
        ], 0.25, "critical")
        domains["risk_management"] = _build_domain([
            hc.check_tail_risk(), hc.check_correlation_regime(),
            hc.check_capital_utilization(),
        ], 0.20, "standard")
        domains["execution_quality"] = _build_domain([
            hc.check_execution_quality(), hc.check_signal_profitability(),
            hc.check_cost_surprise(self.get_execution_quality_history),
        ], 0.20, "standard")
        domains["model_governance"] = _build_domain([
            hc.check_feature_importance_drift(), hc.check_cv_gap_trend(),
            hc.check_regime_transition_health(), hc.check_retraining_effectiveness(),
        ], 0.10, "informational")
        all_cls = [d["checks"] for d in domains.values()]
        checks_total = sum(len(cl) for cl in all_cls)
        checks_avail = sum(1 for cl in all_cls for c in cl if c.get("status") != "UNAVAILABLE")
        overall, overall_status = compute_overall_score(domains)
        domain_ci, overall_ci = compute_confidence_intervals(domains, overall)
        return {
            "overall_score": round(overall, 1) if overall is not None else None,
            "overall_status": overall_status, "overall_ci": overall_ci,
            "overall_methodology": (
                "Weighted average of 5 domains: Data Integrity (25%), Signal Quality (25%), "
                "Risk Management (20%), Execution Quality (20%), Model Governance (10%). "
                "Only checks with available data are scored. Within each domain, checks are "
                "weighted by severity: critical (3x), standard (1x), informational (0.5x). "
                "Confidence intervals computed via bootstrap (N<30) or normal approximation."
            ),
            "checks_available": checks_avail, "checks_total": checks_total,
            "alerts": self._run_alerts(overall, domains),
            "domains": {
                k: {"score": round(v["score"], 1) if v["score"] is not None else None,
                     "weight": v["weight"], "status": v["status"], "ci": domain_ci.get(k),
                     "checks_available": sum(1 for c in v["checks"]
                                             if c.get("status") != "UNAVAILABLE"),
                     "checks_total": len(v["checks"]), "checks": v["checks"]}
                for k, v in domains.items()
            },
        }

    def _run_alerts(self, overall_score, domains):
        try:
            from .health_alerts import create_alert_manager
            mgr = create_alert_manager()
            pending = list(mgr.check_domain_failures(
                {d: v.get("score") for d, v in domains.items() if v.get("score") is not None}))
            if overall_score is not None:
                hist = self.get_health_history(limit=2)
                if hist and hist[-1].get("overall_score") is not None:
                    deg = mgr.check_health_degradation(overall_score, hist[-1]["overall_score"])
                    if deg is not None:
                        pending.append(deg)
            return [a.to_dict() for a in mgr.process_alerts(pending)]
        except Exception as e:
            logger.warning("Alert processing failed: %s", e)
            return []

    @classmethod
    def _resolve_db_path(cls, attr: str) -> Path:
        val = getattr(cls, attr)
        if val is None:
            from quant_engine.config import RESULTS_DIR
            val = Path(RESULTS_DIR) / _DB_FILES[attr]
            setattr(cls, attr, val)
        return val

    @classmethod
    def _get_health_db_path(cls) -> Path: return cls._resolve_db_path("_HEALTH_DB_PATH")
    @classmethod
    def _get_ic_db_path(cls) -> Path: return cls._resolve_db_path("_IC_DB_PATH")
    @classmethod
    def _get_disagreement_db_path(cls) -> Path: return cls._resolve_db_path("_DISAGREEMENT_DB_PATH")
    @classmethod
    def _get_exec_quality_db_path(cls) -> Path: return cls._resolve_db_path("_EXEC_QUALITY_DB_PATH")

    def save_health_snapshot(self, health_result: Dict[str, Any]) -> None:
        from .health_storage import save_health_snapshot
        save_health_snapshot(health_result, max_records=self._MAX_SNAPSHOTS,
                             db_path=self._get_health_db_path())

    def get_health_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        from .health_storage import get_health_history
        return get_health_history(limit=limit, db_path=self._get_health_db_path())

    def get_health_history_with_trends(self, limit: int = 90) -> Dict[str, Any]:
        from .health_storage import get_health_history_with_trends
        return get_health_history_with_trends(limit=limit, db_path=self._get_health_db_path())

    def save_ic_snapshot(self, ic_mean: float, ic_ir: Optional[float] = None,
                         n_candidates: int = 0, n_passed: int = 0,
                         best_strategy_id: str = "") -> None:
        from .health_storage import save_ic_snapshot
        save_ic_snapshot(ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id,
                         max_records=self._MAX_IC_SNAPSHOTS, db_path=self._get_ic_db_path())

    def get_ic_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        from .health_storage import get_ic_history
        return get_ic_history(limit=limit, db_path=self._get_ic_db_path())

    def save_disagreement_snapshot(self, mean_disagreement: float,
                                   max_disagreement: Optional[float] = None, n_members: int = 0,
                                   n_assets: int = 0, pct_high_disagreement: float = 0.0,
                                   member_names: Optional[List[str]] = None) -> None:
        from .health_storage import save_disagreement_snapshot
        save_disagreement_snapshot(mean_disagreement, max_disagreement, n_members,
                                   n_assets, pct_high_disagreement, member_names,
                                   max_records=self._MAX_DISAGREEMENT_SNAPSHOTS,
                                   db_path=self._get_disagreement_db_path())

    def get_disagreement_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        from .health_storage import get_disagreement_history
        return get_disagreement_history(limit=limit, db_path=self._get_disagreement_db_path())

    def save_execution_quality_fill(self, symbol: str, side: str, predicted_cost_bps: float,
                                     actual_cost_bps: float, fill_ratio: Optional[float] = None,
                                     participation_rate: Optional[float] = None,
                                     regime: Optional[int] = None) -> None:
        from .health_storage import save_execution_quality_fill
        save_execution_quality_fill(symbol, side, predicted_cost_bps, actual_cost_bps,
                                     fill_ratio, participation_rate, regime,
                                     max_records=self._MAX_EXEC_QUALITY_RECORDS,
                                     db_path=self._get_exec_quality_db_path())

    def get_execution_quality_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        from .health_storage import get_execution_quality_history
        return get_execution_quality_history(limit=limit, db_path=self._get_exec_quality_db_path())

    @staticmethod
    def _domain_score(checks: List[HealthCheckResult]) -> Optional[float]:
        from .health_scoring import domain_score
        return domain_score(checks)

    @staticmethod
    def _domain_status(checks: List[HealthCheckResult]) -> str:
        from .health_scoring import domain_status
        return domain_status(checks)

    @classmethod
    def _ensure_exec_quality_table(cls) -> None:
        from .health_storage import _ensure_exec_quality_table
        _ensure_exec_quality_table(cls._get_exec_quality_db_path())

    @staticmethod
    def _compute_rolling_average(scores: List[float], window: int = 7) -> List[float]:
        from .health_storage import compute_rolling_average
        return compute_rolling_average(scores, window=window)

    @staticmethod
    def _detect_trend(scores: List[float], window: int = 30) -> tuple:
        from .health_storage import detect_trend
        return detect_trend(scores, window=window)

    @staticmethod
    def _extract_ic_from_cycle_report() -> Optional[float]:
        from .health_checks import _extract_ic_from_cycle_report
        return _extract_ic_from_cycle_report()
