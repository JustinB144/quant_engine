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
        """Read latest backtest summary JSON."""
        from quant_engine.config import RESULTS_DIR

        summary_path = RESULTS_DIR / f"backtest_{horizon}d_summary.json"
        if not summary_path.exists():
            return {"available": False, "horizon": horizon}
        with open(summary_path) as f:
            summary = json.load(f)
        summary["available"] = True
        return summary

    def get_latest_trades(self, horizon: int = 10, limit: int = 200, offset: int = 0) -> Dict[str, Any]:
        """Read trades CSV with pagination."""
        from quant_engine.config import RESULTS_DIR

        import pandas as pd

        path = RESULTS_DIR / f"backtest_{horizon}d_trades.csv"
        if not path.exists():
            return {"available": False, "trades": [], "total": 0}
        df = pd.read_csv(path)
        total = len(df)
        page = df.iloc[offset : offset + limit]
        trades = page.replace({np.nan: None}).to_dict(orient="records")
        return {"available": True, "trades": trades, "total": total, "offset": offset, "limit": limit}

    def get_equity_curve(self, horizon: int = 10) -> Dict[str, Any]:
        """Build equity curve from trade data."""
        from quant_engine.config import RESULTS_DIR
        from quant_engine.dash_ui.data.loaders import build_portfolio_returns, load_trades

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
