"""Wraps backtest results for API consumption."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BacktestService:
    """Reads backtest result files from the results/ directory."""

    def get_latest_results(self, horizon: int = 10) -> Dict[str, Any]:
        """Read latest backtest summary JSON with transparency fields."""
        from quant_engine.config import RESULTS_DIR

        summary_path = RESULTS_DIR / f"backtest_{horizon}d_summary.json"
        if not summary_path.exists():
            return {"available": False, "horizon": horizon}
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            logger.warning("Corrupt backtest summary for horizon %d: %s", horizon, e)
            return {"available": False, "horizon": horizon}
        summary["available"] = True

        # Add model staleness info
        staleness = self._compute_model_staleness()
        summary["model_staleness_days"] = staleness["days"]
        summary["retrain_overdue"] = staleness["overdue"]
        summary["model_version"] = staleness["version_id"]

        # Add sizing method
        summary["sizing_method"] = self._get_sizing_method()

        # Add walk-forward mode
        summary["walk_forward_mode"] = self._get_walk_forward_mode()

        return summary

    def get_latest_trades(self, horizon: int = 10, limit: int = 200, offset: int = 0) -> Dict[str, Any]:
        """Read trades CSV with pagination."""
        from quant_engine.config import RESULTS_DIR

        import pandas as pd

        path = RESULTS_DIR / f"backtest_{horizon}d_trades.csv"
        if not path.exists():
            return {"available": False, "trades": [], "total": 0}
        try:
            df = pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
            logger.warning("Corrupt trades CSV for horizon %d: %s", horizon, e)
            return {"available": False, "trades": [], "total": 0}
        total = len(df)
        page = df.iloc[offset : offset + limit]
        trades = page.replace({np.nan: None}).to_dict(orient="records")
        return {"available": True, "trades": trades, "total": total, "offset": offset, "limit": limit}

    def get_equity_curve(self, horizon: int = 10) -> Dict[str, Any]:
        """Build equity curve from trade data."""
        from quant_engine.config import RESULTS_DIR
        from .data_helpers import build_portfolio_returns, load_trades

        path = RESULTS_DIR / f"backtest_{horizon}d_trades.csv"
        if not path.exists():
            return {"available": False, "points": []}
        trades = load_trades(path)
        returns = build_portfolio_returns(trades)
        if returns.empty:
            return {"available": True, "points": []}
        equity = (1.0 + returns).cumprod()
        # Downsample
        if len(equity) > 2500:
            step = max(1, len(equity) // 2500)
            equity = equity.iloc[::step]
        points = [
            {"date": str(ts.date()) if hasattr(ts, "date") else str(ts), "value": float(v)}
            for ts, v in equity.items()
        ]
        return {"available": True, "points": points}

    def _compute_model_staleness(self) -> Dict[str, Any]:
        """Compute days since last model training."""
        from datetime import datetime, timezone

        try:
            from quant_engine.config import MODEL_DIR
            registry_path = MODEL_DIR / "registry.json"
            if not registry_path.exists():
                return {"days": None, "overdue": False, "version_id": None}
            with open(registry_path) as f:
                reg = json.load(f)
            versions = reg.get("versions", [])
            if not versions:
                return {"days": None, "overdue": False, "version_id": None}
            latest = versions[-1]
            version_id = latest.get("version_id")
            training_date_str = latest.get("training_date")
            if not training_date_str:
                return {"days": None, "overdue": False, "version_id": version_id}
            training_date = datetime.fromisoformat(training_date_str)
            if training_date.tzinfo is None:
                training_date = training_date.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days = (now - training_date).days
            # Default retrain threshold from config
            from quant_engine.config import RETRAIN_MAX_DAYS
            overdue = days > RETRAIN_MAX_DAYS
            return {"days": days, "overdue": overdue, "version_id": version_id}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to compute model staleness: %s", exc)
            return {"days": None, "overdue": False, "version_id": None}

    def _get_sizing_method(self) -> str:
        """Determine the current sizing method from config."""
        try:
            from quant_engine.config import PAPER_USE_KELLY_SIZING
            if PAPER_USE_KELLY_SIZING:
                return "kelly"
        except (ImportError, AttributeError):
            pass
        try:
            from quant_engine.config import EXEC_DYNAMIC_COSTS
            if EXEC_DYNAMIC_COSTS:
                return "dynamic"
        except (ImportError, AttributeError):
            pass
        return "static"

    def _get_walk_forward_mode(self) -> str:
        """Determine the walk-forward mode from config."""
        try:
            from quant_engine.config import WF_MAX_TRAIN_DATES
            if WF_MAX_TRAIN_DATES > 0:
                return "full"
        except (ImportError, AttributeError):
            pass
        return "single_split"
