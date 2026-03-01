"""Execution quality tracking — persistence and retrieval.

Provides ``save_execution_quality_fill`` and ``get_execution_quality_history``
backed by SQLite.  Extracted from api.services so autopilot/paper_trader can
record fill data without importing the API layer (SPEC_AUDIT_FIX_40 T5).
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_EXEC_QUALITY_RECORDS = 500


def _get_exec_quality_db_path() -> Path:
    """Return path to the execution quality SQLite database."""
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "exec_quality.db"


def _ensure_exec_quality_table(db_path: Path) -> None:
    """Create exec_quality table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exec_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                predicted_cost_bps REAL NOT NULL,
                actual_cost_bps REAL NOT NULL,
                cost_surprise_bps REAL NOT NULL,
                fill_ratio REAL,
                participation_rate REAL,
                regime INTEGER
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_execution_quality_fill(
    symbol: str,
    side: str,
    predicted_cost_bps: float,
    actual_cost_bps: float,
    fill_ratio: Optional[float] = None,
    participation_rate: Optional[float] = None,
    regime: Optional[int] = None,
    max_records: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Save a paper-trade fill's execution quality."""
    try:
        path = db_path if db_path is not None else _get_exec_quality_db_path()
        _ensure_exec_quality_table(path)
        cost_surprise = float(actual_cost_bps) - float(predicted_cost_bps)
        conn = sqlite3.connect(str(path))
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO exec_quality "
                "(timestamp, symbol, side, predicted_cost_bps, actual_cost_bps, "
                "cost_surprise_bps, fill_ratio, participation_rate, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp,
                    str(symbol),
                    str(side),
                    float(predicted_cost_bps),
                    float(actual_cost_bps),
                    cost_surprise,
                    float(fill_ratio) if fill_ratio is not None else None,
                    float(participation_rate) if participation_rate is not None else None,
                    int(regime) if regime is not None else None,
                ),
            )
            limit = max_records if max_records is not None else _MAX_EXEC_QUALITY_RECORDS
            conn.execute(
                "DELETE FROM exec_quality WHERE id NOT IN "
                "(SELECT id FROM exec_quality ORDER BY id DESC LIMIT ?)",
                (limit,),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to save execution quality fill: %s", e)


def get_execution_quality_history(
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent execution quality fill records."""
    try:
        path = db_path if db_path is not None else _get_exec_quality_db_path()
        _ensure_exec_quality_table(path)
        conn = sqlite3.connect(str(path))
        try:
            cursor = conn.execute(
                "SELECT timestamp, symbol, side, predicted_cost_bps, "
                "actual_cost_bps, cost_surprise_bps, fill_ratio, "
                "participation_rate, regime "
                "FROM exec_quality ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            history = []
            for (ts, sym, side_val, pred, actual, surprise,
                 fill_r, part_r, reg) in reversed(rows):
                history.append({
                    "timestamp": ts,
                    "symbol": sym,
                    "side": side_val,
                    "predicted_cost_bps": pred,
                    "actual_cost_bps": actual,
                    "cost_surprise_bps": surprise,
                    "fill_ratio": fill_r,
                    "participation_rate": part_r,
                    "regime": reg,
                })
            return history
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to load execution quality history: %s", e)
        return []
