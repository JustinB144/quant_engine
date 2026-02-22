"""
Routing helpers for live vs historical Kalshi endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import pandas as pd


@dataclass
class RouteDecision:
    """Resolved endpoint route decision (base URL, path, and historical/live choice)."""
    base_url: str
    path: str
    use_historical: bool


class KalshiDataRouter:
    """
    Chooses live vs historical endpoint roots by cutoff timestamp.
    """

    def __init__(
        self,
        live_base_url: str,
        historical_base_url: Optional[str] = None,
        historical_cutoff_ts: Optional[str] = None,
        historical_prefix: str = "/historical",
    ):
        """Initialize KalshiDataRouter."""
        self.live_base_url = str(live_base_url).rstrip("/")
        self.historical_base_url = (
            str(historical_base_url).rstrip("/")
            if historical_base_url
            else self.live_base_url
        )
        self.historical_prefix = str(historical_prefix).strip() or "/historical"
        self.historical_cutoff_ts = self._to_utc_ts(historical_cutoff_ts)

    @staticmethod
    def _to_utc_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
        """Internal helper for to utc ts."""
        if value is None:
            return None
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)

    def update_cutoff(self, cutoff_ts: Optional[str]) -> None:
        """Update cutoff in response to input changes."""
        self.historical_cutoff_ts = self._to_utc_ts(cutoff_ts)

    def _extract_end_ts(self, params: Optional[Dict[str, object]]) -> Optional[pd.Timestamp]:
        """Internal helper for extract end ts."""
        p = params or {}
        for key in ("end_ts", "end", "asof_ts"):
            if key in p and p[key] is not None:
                ts = self._to_utc_ts(str(p[key]))
                if ts is not None:
                    return ts
        return None

    @staticmethod
    def _clean_path(path: str) -> str:
        """Internal helper for clean path."""
        raw = str(path).strip() or "/"
        if raw.startswith("http://") or raw.startswith("https://"):
            return urlparse(raw).path or "/"
        return "/" + raw.lstrip("/")

    def resolve(
        self,
        path: str,
        params: Optional[Dict[str, object]] = None,
    ) -> RouteDecision:
        """resolve."""
        clean_path = self._clean_path(path)
        end_ts = self._extract_end_ts(params)
        use_historical = bool(
            self.historical_cutoff_ts is not None
            and end_ts is not None
            and end_ts < self.historical_cutoff_ts
        )

        base_url = self.historical_base_url if use_historical else self.live_base_url
        routed_path = clean_path

        # If historical uses a path partition on same host, add prefix if needed.
        if use_historical and self.historical_base_url == self.live_base_url:
            prefix = "/" + self.historical_prefix.strip("/")
            if not routed_path.startswith(prefix + "/") and routed_path != prefix:
                routed_path = prefix + routed_path

        return RouteDecision(
            base_url=base_url,
            path=routed_path,
            use_historical=use_historical,
        )
