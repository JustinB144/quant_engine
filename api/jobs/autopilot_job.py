"""Autopilot job executor."""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from .runner import JobCancelled


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    """Raise ``JobCancelled`` if the cancellation event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Job cancelled by user")


def execute_autopilot_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run a single autopilot cycle and return results dict."""
    from quant_engine.autopilot.engine import AutopilotEngine

    _check_cancelled(cancel_event)
    if progress_callback:
        progress_callback(0.1, "Starting autopilot cycle")

    engine = AutopilotEngine.from_universe(
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
