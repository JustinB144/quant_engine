"""Wraps models.* modules for API consumption."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelService:
    """Synchronous model metadata / health wrapper."""

    def __init__(self) -> None:
        self._registry = None
        self._governance = None
        try:
            from quant_engine.models.versioning import ModelRegistry
            self._registry = ModelRegistry()
        except Exception as e:
            logger.warning("ModelRegistry unavailable: %s", e)

        try:
            from quant_engine.models.governance import ModelGovernance
            self._governance = ModelGovernance()
        except Exception as e:
            logger.warning("ModelGovernance unavailable: %s", e)

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all registered model versions."""
        if self._registry is None:
            return [{"error": "Model registry unavailable", "status": "degraded"}]
        versions = self._registry.list_versions()
        return [v.to_dict() for v in versions]

    def get_model_health(self) -> Dict[str, Any]:
        """Compute model health from registry + trade data."""
        from quant_engine.config import MODEL_DIR, RESULTS_DIR
        from .data_helpers import compute_model_health, load_trades

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
        from .data_helpers import load_feature_importance

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

    def get_feature_correlations(self, top_n: int = 15) -> Dict[str, Any]:
        """Compute pairwise Pearson correlation between top features using regime importance vectors."""
        from quant_engine.config import MODEL_DIR
        from .data_helpers import load_feature_importance

        empty: Dict[str, Any] = {"feature_names": [], "correlations": [], "n_features": 0}

        global_imp, regime_heat = load_feature_importance(MODEL_DIR)

        if global_imp is None or len(global_imp) == 0:
            return empty
        if regime_heat is None or not hasattr(regime_heat, "corr") or len(regime_heat) == 0:
            return empty

        # regime_heat is features x regimes DataFrame
        # Select top_n features by global importance
        top_features = global_imp.sort_values(ascending=False).head(top_n).index.tolist()

        # Keep only features that exist in the regime heatmap index
        available = [f for f in top_features if f in regime_heat.index]
        if len(available) < 2:
            return empty

        # Subset the regime heatmap to top features and compute correlation
        subset = regime_heat.loc[available]
        corr_matrix = subset.T.corr()

        # Reorder to match the importance ranking
        corr_matrix = corr_matrix.loc[available, available]

        return {
            "feature_names": available,
            "correlations": corr_matrix.values.tolist(),
            "n_features": len(available),
        }

    def get_champion_info(self, horizon: int = 10) -> Dict[str, Any]:
        """Return champion model info for a given horizon."""
        if self._governance is None:
            return {"horizon": horizon, "champion_version": None, "error": "Model governance unavailable", "status": "degraded"}
        version_id = self._governance.get_champion_version(horizon)
        return {
            "horizon": horizon,
            "champion_version": version_id,
        }
