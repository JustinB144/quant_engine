"""
Persistent strategy registry for promoted candidates.
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from ..config import STRATEGY_REGISTRY_PATH, PROMOTION_MAX_ACTIVE_STRATEGIES, PROMOTION_GRACE_CYCLES
from .promotion_gate import PromotionDecision
from ._atomic_write import atomic_json_write, safe_json_load

logger = logging.getLogger(__name__)


@dataclass
class ActiveStrategy:
    """Persisted record for a currently active promoted strategy."""
    strategy_id: str
    promoted_at: str
    params: Dict
    score: float
    metrics: Dict
    status: str = "active"
    consecutive_failures: int = 0

    def to_dict(self) -> Dict:
        """Serialize ActiveStrategy to a dictionary."""
        return asdict(self)


class StrategyRegistry:
    """
    Maintains promoted strategy state and historical promotion decisions.

    Incumbents are protected by a grace period: a strategy must fail
    ``PROMOTION_GRACE_CYCLES`` consecutive cycles before removal,
    preventing churn from transient omissions in candidate sets.
    """

    def __init__(self, path: Path = STRATEGY_REGISTRY_PATH):
        """Initialize StrategyRegistry."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict:
        """Internal helper for load with corrupt-file recovery."""
        return safe_json_load(self.path, {"active": [], "history": []})

    def _save(self, payload: Dict) -> None:
        """Internal helper for atomic save."""
        atomic_json_write(self.path, payload, indent=2)

    def get_active(self) -> List[ActiveStrategy]:
        """Return active."""
        payload = self._load()
        return [ActiveStrategy(**x) for x in payload.get("active", []) if x.get("status") == "active"]

    def apply_promotions(
        self,
        decisions: List[PromotionDecision],
        max_active: int = PROMOTION_MAX_ACTIVE_STRATEGIES,
        grace_cycles: int = PROMOTION_GRACE_CYCLES,
    ) -> List[ActiveStrategy]:
        """Apply promotions with grace-period protection for incumbents.

        Strategies that pass re-evaluation have their failure counter
        reset.  Strategies that fail or are absent from the candidate
        set increment their failure counter.  Only when the counter
        reaches *grace_cycles* is the strategy removed.
        """
        payload = self._load()
        now = datetime.now(timezone.utc).isoformat()

        # Build maps for quick lookup
        passed_map: Dict[str, PromotionDecision] = {}
        failed_set: set = set()
        for d in decisions:
            if d.passed:
                passed_map[d.candidate.strategy_id] = d
            else:
                failed_set.add(d.candidate.strategy_id)

        existing_map: Dict[str, Dict] = {
            x["strategy_id"]: x for x in payload.get("active", [])
        }

        new_active: List[Dict] = []

        # 1) Process incumbents first — keep or increment failure counter
        for sid, row in existing_map.items():
            if sid in passed_map:
                # Incumbent passed re-evaluation: update and reset counter
                d = passed_map[sid]
                row["score"] = float(d.score)
                row["metrics"] = dict(d.metrics)
                row["params"] = d.candidate.to_dict()
                row["status"] = "active"
                row["consecutive_failures"] = 0
                new_active.append(row)
            elif sid in failed_set:
                # Incumbent failed re-evaluation: increment counter
                failures = int(row.get("consecutive_failures", 0)) + 1
                row["consecutive_failures"] = failures
                if failures >= grace_cycles:
                    logger.info(
                        "Removing strategy %s after %d consecutive failures",
                        sid, failures,
                    )
                else:
                    logger.info(
                        "Strategy %s entering grace period (%d/%d failures)",
                        sid, failures, grace_cycles,
                    )
                    new_active.append(row)
            else:
                # Incumbent not in current candidate set: treat as failure
                failures = int(row.get("consecutive_failures", 0)) + 1
                row["consecutive_failures"] = failures
                if failures >= grace_cycles:
                    logger.info(
                        "Removing strategy %s after %d cycles absent from candidates",
                        sid, failures,
                    )
                else:
                    logger.info(
                        "Strategy %s absent from candidates, grace period (%d/%d)",
                        sid, failures, grace_cycles,
                    )
                    new_active.append(row)

        # 2) Add newly promoted strategies (not already incumbent)
        for sid, d in passed_map.items():
            if sid not in existing_map:
                new_active.append(
                    ActiveStrategy(
                        strategy_id=sid,
                        promoted_at=now,
                        params=d.candidate.to_dict(),
                        score=float(d.score),
                        metrics=dict(d.metrics),
                        status="active",
                        consecutive_failures=0,
                    ).to_dict(),
                )

        # 3) Rank by score and cap at max_active
        new_active.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        new_active = new_active[:max_active]

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
