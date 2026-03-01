"""Backtest job executor."""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from .runner import JobCancelled


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    """Raise ``JobCancelled`` if the cancellation event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Job cancelled by user")


def execute_backtest_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run a full backtest pipeline and return results dict."""
    from ..orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    _check_cancelled(cancel_event)
    state = orch.load_and_prepare(
        tickers=params.get("tickers"),
        years=params.get("years", 15),
        feature_mode=params.get("feature_mode", "core"),
        full_universe=params.get("full_universe", False),
        progress_callback=progress_callback,
    )
    _check_cancelled(cancel_event)
    result = orch.backtest(
        state,
        horizon=params.get("horizon", 10),
        version=params.get("version", "latest"),
        risk_management=params.get("risk_management", False),
        holding_period=params.get("holding_period"),
        max_positions=params.get("max_positions"),
        entry_threshold=params.get("entry_threshold"),
        position_size=params.get("position_size"),
        progress_callback=progress_callback,
    )
    return result
