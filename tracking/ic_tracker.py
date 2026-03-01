"""IC (Information Coefficient) tracking — persistence and retrieval.

Provides ``save_ic_snapshot`` and ``get_ic_history`` backed by SQLite.
Extracted from api.services so autopilot can record IC data without
importing the API layer (SPEC_AUDIT_FIX_40 T5).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_IC_SNAPSHOTS = 200


def _get_ic_db_path() -> Path:
    """Return path to the IC tracking SQLite database."""
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "ic_tracking.db"


def _ensure_ic_table(db_path: Path) -> None:
    """Create ic_history table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ic_mean REAL NOT NULL,
                ic_ir REAL,
                n_candidates INTEGER,
                n_passed INTEGER,
                best_strategy_id TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_ic_snapshot(
    ic_mean: float,
    ic_ir: Optional[float] = None,
    n_candidates: int = 0,
    n_passed: int = 0,
    best_strategy_id: str = "",
    max_records: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Save IC metrics from an autopilot cycle."""
    try:
        path = db_path if db_path is not None else _get_ic_db_path()
        _ensure_ic_table(path)
        conn = sqlite3.connect(str(path))
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO ic_history "
                "(timestamp, ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id),
            )
            limit = max_records if max_records is not None else _MAX_IC_SNAPSHOTS
            conn.execute(
                "DELETE FROM ic_history WHERE id NOT IN "
                "(SELECT id FROM ic_history ORDER BY id DESC LIMIT ?)",
                (limit,),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to save IC snapshot: %s", e)


def get_ic_history(
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent IC snapshots."""
    try:
        path = db_path if db_path is not None else _get_ic_db_path()
        _ensure_ic_table(path)
        conn = sqlite3.connect(str(path))
        try:
            cursor = conn.execute(
                "SELECT timestamp, ic_mean, ic_ir, n_candidates, n_passed, best_strategy_id "
                "FROM ic_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            history = []
            for ts, ic_mean_val, ic_ir_val, n_cand, n_pass, strat_id in reversed(rows):
                history.append({
                    "timestamp": ts,
                    "ic_mean": ic_mean_val,
                    "ic_ir": ic_ir_val,
                    "n_candidates": n_cand,
                    "n_passed": n_pass,
                    "best_strategy_id": strat_id,
                })
            return history
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to load IC history: %s", e)
        return []
