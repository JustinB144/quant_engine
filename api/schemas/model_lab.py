"""Model lab schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelVersionInfo(BaseModel):
    """Summary of a model version."""

    version_id: str
    training_date: str
    horizon: int
    universe_size: int
    n_samples: int = 0
    n_features: int = 0
    oos_spearman: float = 0.0
    cv_gap: float = 0.0
    holdout_r2: float = 0.0
    holdout_spearman: float = 0.0
    survivorship_mode: bool = False
    notes: str = ""
    tags: List[str] = Field(default_factory=list)


class ModelHealth(BaseModel):
    """Model health assessment."""

    cv_gap: float = 0.0
    holdout_r2: float = 0.0
    holdout_ic: float = 0.0
    ic_drift: float = 0.0
    retrain_triggered: bool = False
    retrain_reasons: List[str] = Field(default_factory=list)
    registry_history: List[Dict[str, Any]] = Field(default_factory=list)


class FeatureImportance(BaseModel):
    """Feature importance results."""

    global_importance: Dict[str, float] = Field(default_factory=dict)
    regime_heatmap: Dict[str, Any] = Field(default_factory=dict)
