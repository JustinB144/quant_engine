"""
Champion/challenger governance for model versions.
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ..config import CHAMPION_REGISTRY
from ..config import GOVERNANCE_SCORE_WEIGHTS


@dataclass
class ChampionRecord:
    """Persisted champion model record for a prediction horizon."""
    horizon: int
    version_id: str
    promoted_at: str
    score: float
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        """Serialize ChampionRecord to a dictionary."""
        return asdict(self)


class ModelGovernance:
    """
    Maintains champion model per horizon and promotes challengers if better.
    """

    def __init__(self, registry_path: Path = CHAMPION_REGISTRY):
        """Initialize ModelGovernance."""
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict:
        """Internal helper for load."""
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"champions": {}, "history": []}

    def _save(self, payload: Dict):
        """Internal helper for save."""
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    @staticmethod
    def _score(metrics: Dict[str, float]) -> float:
        """Internal helper for score."""
        oos = float(metrics.get("oos_spearman", 0.0))
        hold = float(metrics.get("holdout_spearman", 0.0))
        gap = float(metrics.get("cv_gap", 0.0))
        w = GOVERNANCE_SCORE_WEIGHTS
        return w["oos_spearman"] * oos + w["holdout_spearman"] * hold + w["cv_gap_penalty"] * max(0.0, gap)

    def get_champion_version(self, horizon: int) -> Optional[str]:
        """Return champion version."""
        payload = self._load()
        rec = payload.get("champions", {}).get(str(horizon))
        if rec:
            return rec.get("version_id")
        return None

    def evaluate_and_update(
        self,
        horizon: int,
        version_id: str,
        metrics: Dict[str, float],
        min_relative_improvement: float = 0.02,
    ) -> Dict:
        """Evaluate and update."""
        payload = self._load()
        now = datetime.utcnow().isoformat()
        score = self._score(metrics)

        h_key = str(horizon)
        current = payload.get("champions", {}).get(h_key)
        promoted = False
        reason = "no_current_champion"

        if current is None:
            promoted = True
        else:
            current_score = float(current.get("score", 0.0))
            threshold = current_score * (1.0 + min_relative_improvement)
            if score > threshold:
                promoted = True
                reason = f"score_improved>{min_relative_improvement:.2%}"
            else:
                reason = "challenger_not_better"

        history_row = {
            "timestamp": now,
            "horizon": horizon,
            "version_id": version_id,
            "score": float(score),
            "metrics": metrics,
            "promoted": promoted,
            "reason": reason,
        }
        payload.setdefault("history", []).append(history_row)
        payload["history"] = payload["history"][-2000:]

        if promoted:
            payload.setdefault("champions", {})[h_key] = ChampionRecord(
                horizon=horizon,
                version_id=version_id,
                promoted_at=now,
                score=float(score),
                metrics=metrics,
            ).to_dict()

        self._save(payload)
        return {"promoted": promoted, "reason": reason, "score": float(score)}

