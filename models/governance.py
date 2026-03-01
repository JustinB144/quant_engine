"""
Champion/challenger governance for model versions.
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
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
        """Compute champion score from performance and validation metrics.

        Scoring formula:
            performance (75%): OOS Spearman, holdout Spearman, CV gap penalty
            validation  (25%): DSR significance, PBO, Monte Carlo, calibration ECE

        Validation metrics are optional — when absent, only the performance
        portion contributes. This ensures backward compatibility with models
        trained before validation integration was added.
        """
        # ── Base performance score (existing) ──
        oos = float(metrics.get("oos_spearman", 0.0))
        hold = float(metrics.get("holdout_spearman", 0.0))
        gap = float(metrics.get("cv_gap", 0.0))
        w = GOVERNANCE_SCORE_WEIGHTS
        perf_score = (
            w["oos_spearman"] * oos
            + w["holdout_spearman"] * hold
            + w["cv_gap_penalty"] * max(0.0, gap)
        )

        # ── Validation bonus/penalty (new) ──
        val_score = 0.0

        # DSR significance: bonus for statistical significance, penalty for failure
        if "dsr_significant" in metrics:
            val_score += 0.10 if metrics["dsr_significant"] else -0.10

        # PBO (Probability of Backtest Overfitting): lower = better
        if "pbo" in metrics:
            pbo = float(metrics["pbo"])
            val_score += 0.05 * (1.0 - min(1.0, max(0.0, pbo)))

        # Monte Carlo significance: bonus for statistical significance
        if "mc_significant" in metrics:
            val_score += 0.05 if metrics["mc_significant"] else -0.05

        # Calibration ECE: lower = better calibrated
        if "ece" in metrics and metrics["ece"] is not None:
            ece = float(metrics["ece"])
            val_score += 0.05 * (1.0 - min(1.0, ece * 10))

        return perf_score + val_score

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
        now = datetime.now(timezone.utc).isoformat()
        score = self._score(metrics)

        h_key = str(horizon)
        current = payload.get("champions", {}).get(h_key)
        promoted = False
        reason = "no_current_champion"

        if current is None:
            promoted = True
        else:
            current_score = float(current.get("score", 0.0))
            if current_score >= 0:
                threshold = current_score * (1.0 + min_relative_improvement)
            else:
                # For negative scores, challenger must be better by absolute margin
                # (relative improvement on negative scores inverts the threshold)
                threshold = current_score + abs(current_score) * min_relative_improvement
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

