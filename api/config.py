"""Runtime-adjustable configuration for the API layer."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Keys that may be patched at runtime via the /api/config endpoint.
_ADJUSTABLE_KEYS: Set[str] = {
    "ENTRY_THRESHOLD",
    "CONFIDENCE_THRESHOLD",
    "MAX_POSITIONS",
    "POSITION_SIZE_PCT",
    "REGIME_TRADE_POLICY",
    "DRAWDOWN_WARNING_THRESHOLD",
    "DRAWDOWN_CAUTION_THRESHOLD",
    "DRAWDOWN_CRITICAL_THRESHOLD",
    "DRAWDOWN_DAILY_LOSS_LIMIT",
    "DRAWDOWN_WEEKLY_LOSS_LIMIT",
    "MAX_HOLDING_DAYS",
    "HEALTH_TREND_IMPROVING_THRESHOLD",
    "HEALTH_TREND_DEGRADING_THRESHOLD",
}

# Health trend detection thresholds (SPEC_AUDIT_FIX_04 T3)
HEALTH_TREND_IMPROVING_THRESHOLD: float = 0.5
HEALTH_TREND_DEGRADING_THRESHOLD: float = -0.5

# Semantic validators: key -> (validator_fn, human-readable description).
# Validator returns True if the value is acceptable.
CONFIG_VALIDATORS: Dict[str, tuple[Callable[[Any], bool], str]] = {
    "ENTRY_THRESHOLD": (
        lambda v: 0.0 <= v <= 1.0,
        "Must be between 0.0 and 1.0",
    ),
    "CONFIDENCE_THRESHOLD": (
        lambda v: 0.0 <= v <= 1.0,
        "Must be between 0.0 and 1.0",
    ),
    "MAX_POSITIONS": (
        lambda v: 1 <= v <= 100,
        "Must be between 1 and 100",
    ),
    "POSITION_SIZE_PCT": (
        lambda v: 0.0 < v <= 1.0,
        "Must be between 0.0 (exclusive) and 1.0",
    ),
    "DRAWDOWN_WARNING_THRESHOLD": (
        lambda v: -1.0 <= v <= 0.0,
        "Must be between -1.0 and 0.0",
    ),
    "DRAWDOWN_CAUTION_THRESHOLD": (
        lambda v: -1.0 <= v <= 0.0,
        "Must be between -1.0 and 0.0",
    ),
    "DRAWDOWN_CRITICAL_THRESHOLD": (
        lambda v: -1.0 <= v <= 0.0,
        "Must be between -1.0 and 0.0",
    ),
    "DRAWDOWN_DAILY_LOSS_LIMIT": (
        lambda v: -1.0 <= v <= 0.0,
        "Must be between -1.0 and 0.0",
    ),
    "DRAWDOWN_WEEKLY_LOSS_LIMIT": (
        lambda v: -1.0 <= v <= 0.0,
        "Must be between -1.0 and 0.0",
    ),
    "MAX_HOLDING_DAYS": (
        lambda v: 1 <= v <= 365,
        "Must be between 1 and 365",
    ),
    "HEALTH_TREND_IMPROVING_THRESHOLD": (
        lambda v: 0.0 <= v <= 10.0,
        "Must be between 0.0 and 10.0",
    ),
    "HEALTH_TREND_DEGRADING_THRESHOLD": (
        lambda v: -10.0 <= v <= 0.0,
        "Must be between -10.0 and 0.0",
    ),
}


class ApiSettings(BaseSettings):
    """Immutable settings loaded from environment / .env file."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:8000"
    job_db_path: str = "api_jobs.db"
    log_level: str = "INFO"

    model_config = {"env_prefix": "QE_API_"}


class RuntimeConfig:
    """Thin wrapper around engine ``config.py`` module-level variables.

    Provides get/patch semantics restricted to the adjustable whitelist.
    """

    def __init__(self) -> None:
        import quant_engine.config as _cfg

        self._cfg = _cfg

    def get_adjustable(self) -> Dict[str, Any]:
        """Return the current value of every adjustable key."""
        out: Dict[str, Any] = {}
        for key in sorted(_ADJUSTABLE_KEYS):
            out[key] = getattr(self._cfg, key, None)
        return out

    def patch(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Apply validated updates and return the new state.

        Raises ``KeyError`` for unknown keys.
        """
        bad = set(updates) - _ADJUSTABLE_KEYS
        if bad:
            raise KeyError(f"Keys not adjustable: {sorted(bad)}")
        for key, value in updates.items():
            current = getattr(self._cfg, key)
            # Coerce to same type as current value
            target_type = type(current)
            try:
                if target_type is bool:
                    if isinstance(value, str):
                        coerced = value.lower() in ("true", "1", "yes")
                    else:
                        coerced = bool(value)
                else:
                    coerced = target_type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Cannot coerce {key}={value!r} to {target_type.__name__}") from exc
            # Semantic validation
            validator = CONFIG_VALIDATORS.get(key)
            if validator is not None:
                check_fn, description = validator
                if not check_fn(coerced):
                    raise ValueError(
                        f"Invalid value for {key}: {coerced!r}. {description}"
                    )
            setattr(self._cfg, key, coerced)
            logger.info("RuntimeConfig patched %s = %r", key, coerced)
        return self.get_adjustable()
