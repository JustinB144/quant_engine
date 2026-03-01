"""Ensemble disagreement tracking — persistence and retrieval.

Provides ``save_disagreement_snapshot`` and ``get_disagreement_history``
backed by SQLite.  Extracted from api.services so autopilot can record
disagreement data without importing the API layer (SPEC_AUDIT_FIX_40 T5).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_DISAGREEMENT_SNAPSHOTS = 200


def _get_disagreement_db_path() -> Path:
    """Return path to the disagreement tracking SQLite database."""
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "disagreement_tracking.db"


def _ensure_disagreement_table(db_path: Path) -> None:
    """Create disagreement_history table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS disagreement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mean_disagreement REAL NOT NULL,
                max_disagreement REAL,
                n_members INTEGER,
                n_assets INTEGER,
                pct_high_disagreement REAL,
                member_names TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_disagreement_snapshot(
    mean_disagreement: float,
    max_disagreement: Optional[float] = None,
    n_members: int = 0,
    n_assets: int = 0,
    pct_high_disagreement: float = 0.0,
    member_names: Optional[List[str]] = None,
    max_records: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Save ensemble disagreement metrics from a prediction cycle."""
    try:
        path = db_path if db_path is not None else _get_disagreement_db_path()
        _ensure_disagreement_table(path)
        conn = sqlite3.connect(str(path))
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            names_str = json.dumps(member_names) if member_names else "[]"
            conn.execute(
                "INSERT INTO disagreement_history "
                "(timestamp, mean_disagreement, max_disagreement, n_members, "
                "n_assets, pct_high_disagreement, member_names) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (timestamp, mean_disagreement, max_disagreement,
                 n_members, n_assets, pct_high_disagreement, names_str),
            )
            limit = max_records if max_records is not None else _MAX_DISAGREEMENT_SNAPSHOTS
            conn.execute(
                "DELETE FROM disagreement_history WHERE id NOT IN "
                "(SELECT id FROM disagreement_history ORDER BY id DESC LIMIT ?)",
                (limit,),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to save disagreement snapshot: %s", e)


def get_disagreement_history(
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent disagreement snapshots."""
    try:
        path = db_path if db_path is not None else _get_disagreement_db_path()
        _ensure_disagreement_table(path)
        conn = sqlite3.connect(str(path))
        try:
            cursor = conn.execute(
                "SELECT timestamp, mean_disagreement, max_disagreement, "
                "n_members, n_assets, pct_high_disagreement, member_names "
                "FROM disagreement_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            history = []
            for (ts, mean_d, max_d, n_mem, n_a,
                 pct_high, names_str) in reversed(rows):
                try:
                    member_names_list = json.loads(names_str) if names_str else []
                except (json.JSONDecodeError, TypeError):
                    member_names_list = []
                history.append({
                    "timestamp": ts,
                    "mean_disagreement": mean_d,
                    "max_disagreement": max_d,
                    "n_members": n_mem,
                    "n_assets": n_a,
                    "pct_high_disagreement": pct_high,
                    "member_names": member_names_list,
                })
            return history
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to load disagreement history: %s", e)
        return []
