"""Autopilot job executor."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def execute_autopilot_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run a single autopilot cycle and return results dict."""
    from quant_engine.autopilot.engine import AutopilotEngine

    if progress_callback:
        progress_callback(0.1, "Starting autopilot cycle")

    engine = AutopilotEngine(
        years=params.get("years", 5),
        full_universe=params.get("full_universe", False),
    )
    report = engine.run_cycle()

    if progress_callback:
        progress_callback(1.0, "Autopilot cycle complete")

    # Convert to serialisable dict
    if hasattr(report, "to_dict"):
        return report.to_dict()
    if isinstance(report, dict):
        return report
    return {"status": "completed"}
