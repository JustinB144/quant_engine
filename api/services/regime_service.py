"""Wraps regime.detector for API consumption."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class RegimeService:
    """Synchronous regime detection wrapper."""

    def detect_current_regime(self) -> Dict[str, Any]:
        """Detect current market regime from SPY/benchmark data."""
        from quant_engine.config import DATA_CACHE_DIR, REGIME_NAMES
        from api.services.data_helpers import compute_regime_payload

        from pathlib import Path

        payload = compute_regime_payload(Path(DATA_CACHE_DIR))
        # Convert numpy arrays to lists for JSON serialisation
        trans = payload.get("transition")
        if isinstance(trans, np.ndarray):
            trans = trans.tolist()
        prob_history = payload.get("prob_history")
        history_records = []
        if prob_history is not None and hasattr(prob_history, "iterrows"):
            for ts, row in prob_history.iterrows():
                rec = {"date": str(ts.date()) if hasattr(ts, "date") else str(ts)}
                for col in row.index:
                    rec[col] = float(row[col])
                history_records.append(rec)
                if len(history_records) > 2500:
                    break

        return {
            "current_label": payload.get("current_label", "Unavailable"),
            "as_of": payload.get("as_of", "---"),
            "current_probs": payload.get("current_probs", {}),
            "transition_matrix": trans,
            "prob_history": history_records,
        }

    def get_regime_names(self) -> Dict[int, str]:
        """Return the regime name mapping."""
        from quant_engine.config import REGIME_NAMES

        return dict(REGIME_NAMES)
