"""Reads/writes to the results/ directory."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
        """Read the latest prediction CSV for a horizon.

        Computes cs_zscore (cross-sectional z-score) on the fly if not
        already present, and includes regime_suppressed flag.
        """
        from quant_engine.config import RESULTS_DIR

        import pandas as pd

        path = RESULTS_DIR / f"predictions_{horizon}d.csv"
        if not path.exists():
            return {"available": False, "horizon": horizon}
        df = pd.read_csv(path)

        # Compute cs_zscore if not in the CSV
        if "cs_zscore" not in df.columns and "predicted_return" in df.columns:
            pred_vals = pd.to_numeric(df["predicted_return"], errors="coerce")
            mean_val = pred_vals.mean()
            std_val = pred_vals.std()
            if std_val > 1e-12:
                df["cs_zscore"] = ((pred_vals - mean_val) / std_val).round(4)
            else:
                df["cs_zscore"] = 0.0

        # Add regime_suppressed flag based on per-regime trade policy
        if "regime_suppressed" not in df.columns and "regime" in df.columns:
            try:
                from quant_engine.config import REGIME_TRADE_POLICY
                regime_col = pd.to_numeric(df["regime"], errors="coerce").fillna(-1).astype(int)
                df["regime_suppressed"] = regime_col.map(
                    lambda r: not REGIME_TRADE_POLICY.get(r, {"enabled": True})["enabled"]
                )
            except (ImportError, AttributeError):
                df["regime_suppressed"] = False

        # Replace NaN with None for JSON
        df = df.replace({np.nan: None})
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
