"""Predict job executor."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def execute_predict_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run prediction pipeline and return results dict."""
    from quant_engine.api.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    state = orch.load_and_prepare(
        tickers=params.get("tickers"),
        years=params.get("years", 2),
        feature_mode=params.get("feature_mode", "core"),
        full_universe=params.get("full_universe", False),
        progress_callback=progress_callback,
    )
    result = orch.predict(
        state,
        horizon=params.get("horizon", 10),
        version=params.get("version", "latest"),
        progress_callback=progress_callback,
    )
    return result
