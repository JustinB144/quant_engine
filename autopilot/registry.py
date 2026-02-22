"""
Persistent strategy registry for promoted candidates.
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..config import STRATEGY_REGISTRY_PATH, PROMOTION_MAX_ACTIVE_STRATEGIES
from .promotion_gate import PromotionDecision


@dataclass
class ActiveStrategy:
    """Persisted record for a currently active promoted strategy."""
    strategy_id: str
    promoted_at: str
    params: Dict
    score: float
    metrics: Dict
    status: str = "active"

    def to_dict(self) -> Dict:
        """Serialize ActiveStrategy to a dictionary."""
        return asdict(self)


class StrategyRegistry:
    """
    Maintains promoted strategy state and historical promotion decisions.
    """

    def __init__(self, path: Path = STRATEGY_REGISTRY_PATH):
        """Initialize StrategyRegistry."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict:
        """Internal helper for load."""
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"active": [], "history": []}

    def _save(self, payload: Dict):
        """Internal helper for save."""
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)

    def get_active(self) -> List[ActiveStrategy]:
        """Return active."""
        payload = self._load()
        return [ActiveStrategy(**x) for x in payload.get("active", []) if x.get("status") == "active"]

    def apply_promotions(
        self,
        decisions: List[PromotionDecision],
        max_active: int = PROMOTION_MAX_ACTIVE_STRATEGIES,
    ) -> List[ActiveStrategy]:
        """Apply promotions."""
        payload = self._load()
        now = datetime.utcnow().isoformat()

        ranked = [d for d in decisions if d.passed]
        ranked.sort(key=lambda d: d.score, reverse=True)
        ranked = ranked[:max_active]

        existing_map = {x["strategy_id"]: x for x in payload.get("active", [])}
        new_active: List[Dict] = []

        for d in ranked:
            if d.candidate.strategy_id in existing_map:
                row = existing_map[d.candidate.strategy_id]
                row["score"] = float(d.score)
                row["metrics"] = dict(d.metrics)
                row["params"] = d.candidate.to_dict()
                row["status"] = "active"
                new_active.append(row)
            else:
                new_active.append(
                    ActiveStrategy(
                        strategy_id=d.candidate.strategy_id,
                        promoted_at=now,
                        params=d.candidate.to_dict(),
                        score=float(d.score),
                        metrics=dict(d.metrics),
                        status="active",
                    ).to_dict(),
                )

        payload["active"] = new_active
        for d in decisions:
            payload["history"].append(
                {
                    "timestamp": now,
                    "strategy_id": d.candidate.strategy_id,
                    "passed": d.passed,
                    "score": float(d.score),
                    "reasons": d.reasons,
                    "metrics": d.metrics,
                    "params": d.candidate.to_dict(),
                },
            )

        # Keep history bounded.
        payload["history"] = payload["history"][-2000:]
        self._save(payload)
        return [ActiveStrategy(**x) for x in new_active]

