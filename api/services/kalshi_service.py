"""Wraps kalshi.storage for API consumption."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class KalshiService:
    """Kalshi event market data — conditionally enabled."""

    def __init__(self) -> None:
        from quant_engine.config import KALSHI_ENABLED

        self.enabled = KALSHI_ENABLED

    def get_events(self, event_type: str | None = None) -> Dict[str, Any]:
        """Query stored Kalshi events."""
        if not self.enabled:
            return {"enabled": False, "events": []}
        try:
            from quant_engine.config import KALSHI_DB_PATH
            from quant_engine.kalshi.storage import EventTimeStore

            store = EventTimeStore(KALSHI_DB_PATH)
            df = store.query_df(
                "SELECT * FROM kalshi_markets LIMIT 200"
            )
            if event_type:
                df = df[df["event_type"].str.contains(event_type, case=False, na=False)]
            records = df.head(200).to_dict(orient="records")
            return {"enabled": True, "events": records, "total": len(df)}
        except Exception as exc:
            logger.warning("Kalshi query failed: %s", exc)
            return {"enabled": True, "events": [], "error": str(exc)}

    def get_distributions(self, market_id: str) -> Dict[str, Any]:
        """Return contract-level distribution for a market."""
        if not self.enabled:
            return {"enabled": False}
        try:
            from quant_engine.config import KALSHI_DB_PATH
            from quant_engine.kalshi.storage import EventTimeStore

            store = EventTimeStore(KALSHI_DB_PATH)
            df = store.query_df(
                "SELECT * FROM kalshi_contracts WHERE event_id = ?",
                params=[market_id],
            )
            records = df.to_dict(orient="records") if len(df) else []
            return {"enabled": True, "market_id": market_id, "contracts": records}
        except Exception as exc:
            logger.warning("Kalshi distribution query failed: %s", exc)
            return {"enabled": True, "market_id": market_id, "contracts": [], "error": str(exc)}
