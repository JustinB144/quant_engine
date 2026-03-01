"""Train job executor."""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from .runner import JobCancelled


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    """Raise ``JobCancelled`` if the cancellation event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Job cancelled by user")


def execute_train_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run a full train pipeline and return results dict."""
    from ..orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    _check_cancelled(cancel_event)
    state = orch.load_and_prepare(
        tickers=params.get("tickers"),
        years=params.get("years", 5),
        feature_mode=params.get("feature_mode", "core"),
        survivorship_mode=params.get("survivorship", False),
        full_universe=params.get("full_universe", False),
        progress_callback=progress_callback,
    )
    _check_cancelled(cancel_event)
    result = orch.train(
        state,
        horizons=params.get("horizons", [10]),
        survivorship_mode=params.get("survivorship", False),
        recency_weight=params.get("recency", False),
        progress_callback=progress_callback,
    )
    return result
