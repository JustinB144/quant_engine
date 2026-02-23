"""System health assessment for API consumption."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class HealthService:
    """Computes system health from model age, cache freshness, etc."""

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

        return {"status": status, "checks": checks, "timestamp": datetime.now(timezone.utc).isoformat()}

    def get_detailed_health(self) -> Dict[str, Any]:
        """Full system health assessment."""
        from api.services.data_helpers import collect_health_data

        payload = collect_health_data()
        # Convert dataclass to dict, handling nested dataclasses
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
        # Convert HealthCheck lists to dicts
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
        return result
