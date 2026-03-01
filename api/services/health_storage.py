"""Health history SQLite storage — snapshot persistence, retrieval, and trends.

Includes storage for:
- Health history snapshots (health_history.db)
- IC tracking (ic_tracking.db)
- Ensemble disagreement tracking (disagreement_tracking.db)
- Execution quality tracking (exec_quality.db)

Extracted from HealthService (SPEC_AUDIT_FIX_04 T5) for independent testability.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_SNAPSHOTS = 90  # 90 days retention (Spec 09)
_MAX_IC_SNAPSHOTS = 200
_MAX_DISAGREEMENT_SNAPSHOTS = 200
_MAX_EXEC_QUALITY_RECORDS = 500


def _get_health_db_path() -> Path:
    """Return the path to the health history SQLite database."""
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "health_history.db"


def _ensure_health_table(db_path: Path) -> None:
    """Create health_history table if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_score REAL,
                domain_scores TEXT,
                check_results TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_health_snapshot(health_result: Dict[str, Any],
                         max_records: Optional[int] = None,
                         db_path: Optional[Path] = None) -> None:
    """Save a health check snapshot to the SQLite database."""
    try:
        path = db_path if db_path is not None else _get_health_db_path()
        _ensure_health_table(path)
        conn = sqlite3.connect(str(path))
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            overall_score = health_result.get("overall_score")
            domain_scores = json.dumps({
                k: {"score": v.get("score"), "status": v.get("status")}
                for k, v in health_result.get("domains", {}).items()
            })
            check_results = json.dumps(health_result)

            conn.execute(
                "INSERT INTO health_history (timestamp, overall_score, domain_scores, check_results) "
                "VALUES (?, ?, ?, ?)",
                (timestamp, overall_score, domain_scores, check_results),
            )

            # Prune old entries beyond max
            limit = max_records if max_records is not None else _MAX_SNAPSHOTS
            conn.execute(
                "DELETE FROM health_history WHERE id NOT IN "
                "(SELECT id FROM health_history ORDER BY id DESC LIMIT ?)",
                (limit,),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to save health snapshot: %s", e)


def get_health_history(limit: int = 30,
                       db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve recent health snapshots for trend visualization."""
    try:
        path = db_path if db_path is not None else _get_health_db_path()
        _ensure_health_table(path)
        conn = sqlite3.connect(str(path))
        try:
            cursor = conn.execute(
                "SELECT timestamp, overall_score, domain_scores "
                "FROM health_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            history = []
            for ts, score, domain_json in reversed(rows):
                entry: Dict[str, Any] = {
                    "timestamp": ts,
                    "overall_score": score,
                }
                try:
                    entry["domain_scores"] = json.loads(domain_json)
                except (json.JSONDecodeError, TypeError):
                    entry["domain_scores"] = {}
                history.append(entry)
            return history
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to load health history: %s", e)
        return []


def compute_rolling_average(
    scores: List[float],
    window: int = 7,
) -> List[float]:
    """Compute rolling average of health scores."""
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    if len(arr) < window:
        window = len(arr)
    if window <= 0:
        return scores
    cumsum = np.cumsum(arr)
    cumsum = np.insert(cumsum, 0, 0.0)
    rolling = (cumsum[window:] - cumsum[:-window]) / window
    # Pad front with partial averages
    pad = [float(np.mean(arr[:i + 1])) for i in range(min(window - 1, len(arr)))]
    return pad + rolling.tolist()


def detect_trend(
    scores: List[float],
    window: int = 30,
) -> tuple:
    """Detect health score trend using linear regression.

    Returns
    -------
    (trend_label, slope)
    """
    if len(scores) < 3:
        return ("unknown", 0.0)

    recent = scores[-window:] if len(scores) >= window else scores
    x = np.arange(len(recent), dtype=float)
    coeffs = np.polyfit(x, recent, 1)
    slope = float(coeffs[0])

    try:
        from quant_engine.api.config import (
            HEALTH_TREND_IMPROVING_THRESHOLD,
            HEALTH_TREND_DEGRADING_THRESHOLD,
        )
    except ImportError:
        HEALTH_TREND_IMPROVING_THRESHOLD = 0.5
        HEALTH_TREND_DEGRADING_THRESHOLD = -0.5

    if slope > HEALTH_TREND_IMPROVING_THRESHOLD:
        trend = "improving"
    elif slope < HEALTH_TREND_DEGRADING_THRESHOLD:
        trend = "degrading"
    else:
        trend = "stable"

    return (trend, slope)


def get_health_history_with_trends(limit: int = 90,
                                   db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve health history with rolling averages and trend analysis."""
    history = get_health_history(limit=limit, db_path=db_path)

    if not history:
        return {
            "snapshots": [],
            "rolling_7d": [],
            "rolling_30d": [],
            "trend": "unknown",
            "trend_slope": 0.0,
        }

    scores = [h.get("overall_score") for h in history]
    valid_scores = [s for s in scores if s is not None]

    rolling_7d = compute_rolling_average(valid_scores, window=7)
    rolling_30d = compute_rolling_average(valid_scores, window=30)
    trend, slope = detect_trend(valid_scores, window=30)

    return {
        "snapshots": history,
        "rolling_7d": [round(v, 1) for v in rolling_7d],
        "rolling_30d": [round(v, 1) for v in rolling_30d],
        "trend": trend,
        "trend_slope": round(slope, 4),
    }


# ── IC Tracking Storage (SPEC-H01) ────────────────────────────────────

def _get_ic_db_path() -> Path:
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "ic_tracking.db"


def _ensure_ic_table(db_path: Path) -> None:
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


def get_ic_history(limit: int = 20,
                   db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
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
            for ts, ic_mean, ic_ir, n_cand, n_pass, strat_id in reversed(rows):
                history.append({
                    "timestamp": ts,
                    "ic_mean": ic_mean,
                    "ic_ir": ic_ir,
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


# ── Disagreement Tracking Storage (SPEC-H02) ──────────────────────────

def _get_disagreement_db_path() -> Path:
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "disagreement_tracking.db"


def _ensure_disagreement_table(db_path: Path) -> None:
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


def get_disagreement_history(limit: int = 20,
                             db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
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


# ── Execution Quality Storage (SPEC-H03) ──────────────────────────────

def _get_exec_quality_db_path() -> Path:
    from quant_engine.config import RESULTS_DIR
    return Path(RESULTS_DIR) / "exec_quality.db"


def _ensure_exec_quality_table(db_path: Path) -> None:
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
            for (ts, sym, side, pred, actual, surprise,
                 fill_r, part_r, reg) in reversed(rows):
                history.append({
                    "timestamp": ts,
                    "symbol": sym,
                    "side": side,
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
