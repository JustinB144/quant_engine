"""Wraps kalshi.storage for API consumption."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class KalshiService:
    """Kalshi event market data — conditionally enabled."""

    def __init__(self) -> None:
        from quant_engine.config import KALSHI_ENABLED

        self.enabled = KALSHI_ENABLED

    def get_events(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        """Query stored Kalshi events."""
        if not self.enabled:
            return {"enabled": False, "events": []}
        try:
            from quant_engine.config import KALSHI_DB_PATH
            from quant_engine.kalshi.storage import EventTimeStore

            with EventTimeStore(KALSHI_DB_PATH) as store:
                if event_type:
                    df = store.query_df(
                        "SELECT * FROM kalshi_markets WHERE event_type LIKE ? LIMIT 200",
                        params=[f"%{event_type}%"],
                    )
                else:
                    df = store.query_df("SELECT * FROM kalshi_markets LIMIT 200")
                records = df.to_dict(orient="records")
            return {"enabled": True, "events": records, "total": len(records)}
        except Exception as exc:
            logger.warning("Kalshi query failed: %s", exc)
            return {"enabled": True, "events": [], "error": str(exc)}

    def get_distributions(self, market_id: str) -> Dict[str, Any]:
        """Return distribution-level data for a market."""
        if not self.enabled:
            return {"enabled": False}
        try:
            from quant_engine.config import KALSHI_DB_PATH
            from quant_engine.kalshi.storage import EventTimeStore

            with EventTimeStore(KALSHI_DB_PATH) as store:
                df = store.query_df(
                    "SELECT * FROM kalshi_distributions WHERE market_id = ?",
                    params=[market_id],
                )
                records = df.to_dict(orient="records") if len(df) else []
            return {"enabled": True, "market_id": market_id, "distributions": records}
        except Exception as exc:
            logger.warning("Kalshi distribution query failed: %s", exc)
            return {"enabled": True, "market_id": market_id, "distributions": [], "error": str(exc)}
