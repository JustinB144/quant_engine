"""
Versioned event-to-market mapping persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .storage import EventTimeStore


@dataclass
class EventMarketMappingRecord:
    """Versioned mapping row linking a macro event to a Kalshi market over an effective time window."""
    event_id: str
    market_id: str
    event_type: str = ""
    effective_start_ts: str = ""
    effective_end_ts: Optional[str] = None
    mapping_version: str = "v1"
    source: str = "manual"


class EventMarketMappingStore:
    """Persistence helper for versioned event-to-market mappings stored in EventTimeStore."""
    def __init__(self, store: EventTimeStore):
        """Initialize EventMarketMappingStore."""
        self.store = store

    def upsert(self, rows: Iterable[EventMarketMappingRecord | dict]) -> None:
        """upsert."""
        payload = []
        for row in rows:
            if isinstance(row, EventMarketMappingRecord):
                payload.append(
                    {
                        "event_id": row.event_id,
                        "event_type": row.event_type,
                        "market_id": row.market_id,
                        "effective_start_ts": row.effective_start_ts,
                        "effective_end_ts": row.effective_end_ts,
                        "mapping_version": row.mapping_version,
                        "source": row.source,
                    },
                )
            else:
                payload.append(dict(row))
        self.store.upsert_event_market_map_versions(payload)

    def asof(self, asof_ts: str) -> pd.DataFrame:
        """asof."""
        return self.store.get_event_market_map_asof(asof_ts)

    def current_version(self) -> str:
        """Return the latest mapping_version string from the store."""
        df = self.store.get_event_market_map_asof(
            pd.Timestamp.now(tz="UTC").isoformat()
        )
        if df.empty or "mapping_version" not in df.columns:
            return "v1"
        return str(df["mapping_version"].iloc[-1])

    @staticmethod
    def assert_consistent_mapping_version(panel: pd.DataFrame) -> None:
        """Raise ValueError if a panel contains mixed mapping versions (D1)."""
        if "mapping_version" not in panel.columns:
            return
        versions = panel["mapping_version"].dropna().unique()
        if len(versions) > 1:
            raise ValueError(
                f"Panel contains mixed mapping versions: {sorted(versions)}. "
                "Merging data across mapping versions is unsafe."
            )
