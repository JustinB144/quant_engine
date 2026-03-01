"""Standard API response envelope with provenance metadata."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseMeta(BaseModel):
    """Provenance metadata attached to every API response."""

    data_mode: str = "live"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: List[str] = Field(default_factory=list)
    source_summary: Optional[str] = None
    predictor_type: Optional[str] = None
    walk_forward_mode: Optional[str] = None
    regime_suppressed: Optional[bool] = None
    regime_trade_policy: Optional[str] = None
    feature_pipeline_version: Optional[str] = None
    model_version: Optional[str] = None
    sizing_method: Optional[str] = None
    cache_hit: bool = False
    elapsed_ms: Optional[float] = None


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""

    ok: bool = True
    data: Optional[T] = None
    error: Optional[str] = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @classmethod
    def success(cls, data: Any, *, meta: Optional[ResponseMeta] = None, **meta_kwargs) -> "ApiResponse":
        """Build a success response."""
        if meta is None:
            meta = ResponseMeta(**meta_kwargs)
        return cls(ok=True, data=data, meta=meta)

    @classmethod
    def fail(cls, error: str, *, warnings: Optional[List[str]] = None) -> "ApiResponse":
        """Build an error response."""
        meta = ResponseMeta(warnings=warnings or [])
        return cls(ok=False, error=error, meta=meta)

    @classmethod
    def from_cached(cls, data: Any, elapsed_ms: float = 0.0, **meta_kwargs) -> "ApiResponse":
        """Build a response flagged as served from cache."""
        meta = ResponseMeta(cache_hit=True, elapsed_ms=elapsed_ms, **meta_kwargs)
        return cls(ok=True, data=data, meta=meta)
