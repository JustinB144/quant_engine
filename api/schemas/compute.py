"""Request schemas for compute (POST) endpoints."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Request body for POST /api/models/train."""

    horizons: List[int] = Field(default=[10])
    tickers: Optional[List[str]] = None
    years: int = 5
    feature_mode: str = "core"
    survivorship: bool = False
    full_universe: bool = False
    recency: bool = False


class BacktestRequest(BaseModel):
    """Request body for POST /api/backtests/run."""

    horizon: int = 10
    tickers: Optional[List[str]] = None
    years: int = 15
    feature_mode: str = "core"
    risk_management: bool = False
    version: str = "latest"
    full_universe: bool = False


class PredictRequest(BaseModel):
    """Request body for POST /api/models/predict."""

    horizon: int = 10
    tickers: Optional[List[str]] = None
    years: int = 2
    feature_mode: str = "core"
    version: str = "latest"
    full_universe: bool = False


class AutopilotRequest(BaseModel):
    """Request body for POST /api/autopilot/run-cycle."""

    years: int = 5
    full_universe: bool = False


class JobCreatedResponse(BaseModel):
    """Response for job submission endpoints."""

    job_id: str
    job_type: str
    status: str = "queued"
