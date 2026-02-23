"""Wraps models.* modules for API consumption."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelService:
    """Synchronous model metadata / health wrapper."""

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all registered model versions."""
        from quant_engine.models.versioning import ModelRegistry

        registry = ModelRegistry()
        versions = registry.list_versions()
        return [v.to_dict() for v in versions]

    def get_model_health(self) -> Dict[str, Any]:
        """Compute model health from registry + trade data."""
        from quant_engine.config import MODEL_DIR, RESULTS_DIR
        from quant_engine.dash_ui.data.loaders import compute_model_health, load_trades

        trades_path = RESULTS_DIR / "backtest_10d_trades.csv"
        trades = load_trades(trades_path)
        health = compute_model_health(MODEL_DIR, trades)
        # Convert registry_history DataFrame to list of dicts
        reg_hist = health.pop("registry_history", None)
        if reg_hist is not None and hasattr(reg_hist, "to_dict"):
            health["registry_history"] = reg_hist.to_dict(orient="records")
        return health

    def get_feature_importance(self) -> Dict[str, Any]:
        """Load feature importance from latest model meta."""
        from quant_engine.config import MODEL_DIR
        from quant_engine.dash_ui.data.loaders import load_feature_importance

        global_imp, regime_heat = load_feature_importance(MODEL_DIR)
        result: Dict[str, Any] = {
            "global_importance": {},
            "regime_heatmap": {},
        }
        if global_imp is not None and len(global_imp) > 0:
            result["global_importance"] = global_imp.sort_values(ascending=False).head(30).to_dict()
        if regime_heat is not None and hasattr(regime_heat, "to_dict") and len(regime_heat) > 0:
            result["regime_heatmap"] = regime_heat.to_dict()
        return result

    def get_champion_info(self, horizon: int = 10) -> Dict[str, Any]:
        """Return champion model info for a given horizon."""
        from quant_engine.models.governance import ModelGovernance

        gov = ModelGovernance()
        version_id = gov.get_champion_version(horizon)
        return {
            "horizon": horizon,
            "champion_version": version_id,
        }
