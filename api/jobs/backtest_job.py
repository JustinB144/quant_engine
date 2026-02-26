"""Backtest job executor."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def execute_backtest_job(
    params: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run a full backtest pipeline and return results dict."""
    from ..orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    state = orch.load_and_prepare(
        tickers=params.get("tickers"),
        years=params.get("years", 15),
        feature_mode=params.get("feature_mode", "core"),
        full_universe=params.get("full_universe", False),
        progress_callback=progress_callback,
    )
    result = orch.backtest(
        state,
        horizon=params.get("horizon", 10),
        version=params.get("version", "latest"),
        risk_management=params.get("risk_management", False),
        progress_callback=progress_callback,
    )
    return result
