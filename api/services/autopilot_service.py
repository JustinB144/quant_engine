"""Wraps autopilot engine and results for API consumption."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AutopilotService:
    """Reads autopilot state and cycle reports."""

    def get_latest_cycle(self) -> Dict[str, Any]:
        """Read the most recent autopilot cycle report."""
        from quant_engine.config import AUTOPILOT_CYCLE_REPORT

        path = AUTOPILOT_CYCLE_REPORT
        if not path.exists():
            return {"available": False}
        try:
            with open(path) as f:
                report = json.load(f)
            report["available"] = True
            return report
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cycle report: %s", exc)
            return {"available": False, "error": str(exc)}

    def get_strategy_registry(self) -> Dict[str, Any]:
        """Return active and historical strategies."""
        from quant_engine.config import STRATEGY_REGISTRY_PATH

        path = STRATEGY_REGISTRY_PATH
        if not path.exists():
            return {"active": [], "history_count": 0}
        try:
            with open(path) as f:
                payload = json.load(f)
            return {
                "active": payload.get("active", []),
                "history_count": len(payload.get("history", [])),
            }
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read strategy registry: %s", exc)
            return {"active": [], "history_count": 0, "error": str(exc)}

    def get_paper_state(self) -> Dict[str, Any]:
        """Return current paper-trading state."""
        from quant_engine.config import PAPER_STATE_PATH

        path = PAPER_STATE_PATH
        if not path.exists():
            return {"available": False}
        try:
            with open(path) as f:
                state = json.load(f)
            state["available"] = True
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read paper state: %s", exc)
            return {"available": False, "error": str(exc)}
