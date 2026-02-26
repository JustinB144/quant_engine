"""Runtime-adjustable configuration for the API layer."""
from __future__ import annotations

import logging
from typing import Any, Dict, Set

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
}


class ApiSettings(BaseSettings):
    """Immutable settings loaded from environment / .env file."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
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
                coerced = target_type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Cannot coerce {key}={value!r} to {target_type.__name__}") from exc
            setattr(self._cfg, key, coerced)
            logger.info("RuntimeConfig patched %s = %r", key, coerced)
        return self.get_adjustable()
