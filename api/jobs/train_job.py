"""Train job executor."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def execute_train_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run a full train pipeline and return results dict."""
    from quant_engine.api.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    state = orch.load_and_prepare(
        tickers=params.get("tickers"),
        years=params.get("years", 5),
        feature_mode=params.get("feature_mode", "core"),
        survivorship_mode=params.get("survivorship", False),
        full_universe=params.get("full_universe", False),
        progress_callback=progress_callback,
    )
    result = orch.train(
        state,
        horizons=params.get("horizons", [10]),
        survivorship_mode=params.get("survivorship", False),
        recency_weight=params.get("recency", False),
        progress_callback=progress_callback,
    )
    return result
