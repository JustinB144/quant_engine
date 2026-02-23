"""Route modules — imported lazily by the app factory."""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Module paths that provide a ``router`` attribute.
_ROUTER_MODULES = [
    "quant_engine.api.routers.jobs",
    "quant_engine.api.routers.system_health",
    "quant_engine.api.routers.dashboard",
    "quant_engine.api.routers.data_explorer",
    "quant_engine.api.routers.model_lab",
    "quant_engine.api.routers.signals",
    "quant_engine.api.routers.backtests",
    "quant_engine.api.routers.benchmark",
    "quant_engine.api.routers.logs",
    "quant_engine.api.routers.autopilot",
    "quant_engine.api.routers.config_mgmt",
    "quant_engine.api.routers.iv_surface",
]


def all_routers() -> List[APIRouter]:
    """Import and return every available router, skipping broken ones."""
    import importlib

    routers: List[APIRouter] = []
    for mod_path in _ROUTER_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            routers.append(mod.router)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping router %s: %s", mod_path, exc)
    return routers
