"""Reads/writes to the results/ directory."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResultsService:
    """Unified access to persisted result artefacts."""

    def get_latest_backtest(self, horizon: int = 10) -> Dict[str, Any]:
        """Read the latest backtest summary for a horizon."""
        from quant_engine.config import RESULTS_DIR

        path = RESULTS_DIR / f"backtest_{horizon}d_summary.json"
        if not path.exists():
            return {"available": False, "horizon": horizon}
        with open(path) as f:
            data = json.load(f)
        data["available"] = True
        return data

    def get_latest_predictions(self, horizon: int = 10) -> Dict[str, Any]:
        """Read the latest prediction CSV for a horizon."""
        from quant_engine.config import RESULTS_DIR

        import pandas as pd

        path = RESULTS_DIR / f"predictions_{horizon}d.csv"
        if not path.exists():
            return {"available": False, "horizon": horizon}
        df = pd.read_csv(path)
        records = df.head(200).to_dict(orient="records")
        return {"available": True, "horizon": horizon, "signals": records, "total": len(df)}

    def list_all_results(self) -> List[Dict[str, Any]]:
        """List all result files in the results/ directory."""
        from quant_engine.config import RESULTS_DIR

        results_dir = Path(RESULTS_DIR)
        if not results_dir.exists():
            return []
        items = []
        for p in sorted(results_dir.glob("*")):
            if p.is_file():
                items.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "modified": p.stat().st_mtime,
                })
        return items
