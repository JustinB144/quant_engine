"""
Feature Stability Monitoring — tracks feature importance rankings across
training cycles and alerts on dramatic distributional shifts.

Usage::

    tracker = FeatureStabilityTracker()
    tracker.record_importance("cycle_001", importances_array, feature_names)
    report = tracker.check_stability()

Storage is a JSON file at ``RESULTS_DIR / "feature_stability_history.json"``.
"""
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
except ImportError:  # pragma: no cover - optional dependency fallback
    sp_stats = None

from ..config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of top features to track per cycle.
TOP_N: int = 30

#: Spearman correlation threshold below which a shift is considered dramatic.
SHIFT_THRESHOLD: float = 0.50

#: Default storage path.
DEFAULT_HISTORY_PATH: Path = RESULTS_DIR / "feature_stability_history.json"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    """Summary returned by :meth:`FeatureStabilityTracker.check_stability`."""

    n_cycles: int
    latest_cycle_id: Optional[str]
    spearman_vs_previous: Optional[float]
    spearman_vs_first: Optional[float]
    mean_spearman: Optional[float]
    alert: bool
    alert_message: str
    pairwise_correlations: List[Dict]
    top_features_latest: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize StabilityReport to a dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class FeatureStabilityTracker:
    """Record and compare feature importance rankings over training cycles.

    Parameters
    ----------
    history_path:
        Path to the JSON file where cycle history is persisted.
        Defaults to ``RESULTS_DIR / "feature_stability_history.json"``.
    top_n:
        Number of top features to retain per cycle (ranked by importance).
    shift_threshold:
        Spearman rank correlation below this value triggers an alert.
    """

    def __init__(
        self,
        history_path: Optional[Path] = None,
        top_n: int = TOP_N,
        shift_threshold: float = SHIFT_THRESHOLD,
    ):
        """Initialize FeatureStabilityTracker."""
        self.history_path = Path(history_path) if history_path else DEFAULT_HISTORY_PATH
        self.top_n = top_n
        self.shift_threshold = shift_threshold

        # In-memory history: list of cycle records.
        self._history: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing history from disk, if present."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self._history = data
                elif isinstance(data, dict) and "cycles" in data:
                    self._history = data["cycles"]
                else:
                    self._history = []
            except (json.JSONDecodeError, OSError):
                self._history = []

    def _save(self) -> None:
        """Persist current history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cycles": self._history,
        }
        with open(self.history_path, "w") as fh:
            json.dump(payload, fh, indent=2)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_importance(
        self,
        cycle_id: str,
        feature_importances: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """Store the top-N feature importance ranking for a training cycle.

        Parameters
        ----------
        cycle_id:
            Unique identifier for this training cycle (e.g. a timestamp
            or version string).
        feature_importances:
            Array of importance values aligned with *feature_names*.
        feature_names:
            Names of the features corresponding to *feature_importances*.
        """
        importances = np.asarray(feature_importances, dtype=float)
        if len(importances) != len(feature_names):
            raise ValueError(
                f"feature_importances length ({len(importances)}) != "
                f"feature_names length ({len(feature_names)})"
            )

        # Sort descending by importance and keep top N.
        order = np.argsort(importances)[::-1]
        top_n = min(self.top_n, len(order))
        top_idx = order[:top_n]

        record: Dict = {
            "cycle_id": cycle_id,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "top_features": [feature_names[i] for i in top_idx],
            "top_importances": [float(importances[i]) for i in top_idx],
            "total_features": len(feature_names),
        }
        self._history.append(record)
        self._save()

    # ------------------------------------------------------------------
    # Stability analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _spearman_rank_correlation(
        ranking_a: List[str],
        importances_a: List[float],
        ranking_b: List[str],
        importances_b: List[float],
    ) -> float:
        """Compute Spearman rank correlation between two importance rankings.

        Features present in only one ranking are assigned the worst rank
        in the other (rank = len + 1) so that novel/dropped features
        naturally reduce correlation.
        """
        # Build union of features across both rankings.
        all_features = list(dict.fromkeys(ranking_a + ranking_b))

        # Assign ranks (1-based; missing features get len+1).
        rank_a_map = {f: r + 1 for r, f in enumerate(ranking_a)}
        rank_b_map = {f: r + 1 for r, f in enumerate(ranking_b)}

        default_rank_a = len(ranking_a) + 1
        default_rank_b = len(ranking_b) + 1

        ranks_a = np.array([rank_a_map.get(f, default_rank_a) for f in all_features], dtype=float)
        ranks_b = np.array([rank_b_map.get(f, default_rank_b) for f in all_features], dtype=float)

        if len(all_features) < 2:
            return 1.0

        # Use scipy if available, otherwise manual calculation.
        if sp_stats is not None:
            corr, _ = sp_stats.spearmanr(ranks_a, ranks_b)
            return float(corr) if not np.isnan(corr) else 0.0

        # Manual Spearman: correlation of ranks.
        ra = ranks_a - ranks_a.mean()
        rb = ranks_b - ranks_b.mean()
        denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(ra * rb) / denom)

    def check_stability(self) -> StabilityReport:
        """Analyse the recorded history and return a stability report.

        The report includes:
        - Spearman rank correlation between the latest and previous cycle.
        - Spearman rank correlation between the latest and first cycle.
        - Mean pairwise Spearman across all consecutive cycles.
        - An alert flag and message if the latest shift exceeds the
          threshold.

        Returns
        -------
        StabilityReport
        """
        n = len(self._history)
        if n == 0:
            return StabilityReport(
                n_cycles=0,
                latest_cycle_id=None,
                spearman_vs_previous=None,
                spearman_vs_first=None,
                mean_spearman=None,
                alert=False,
                alert_message="No cycles recorded yet.",
                pairwise_correlations=[],
            )

        if n == 1:
            latest = self._history[-1]
            return StabilityReport(
                n_cycles=1,
                latest_cycle_id=latest["cycle_id"],
                spearman_vs_previous=None,
                spearman_vs_first=None,
                mean_spearman=None,
                alert=False,
                alert_message="Only one cycle recorded; stability comparison requires at least two.",
                pairwise_correlations=[],
                top_features_latest=latest["top_features"],
            )

        # Compute pairwise Spearman for all consecutive pairs.
        pairwise: List[Dict] = []
        for i in range(1, n):
            prev = self._history[i - 1]
            curr = self._history[i]
            corr = self._spearman_rank_correlation(
                prev["top_features"], prev["top_importances"],
                curr["top_features"], curr["top_importances"],
            )
            pairwise.append({
                "from_cycle": prev["cycle_id"],
                "to_cycle": curr["cycle_id"],
                "spearman": round(corr, 4),
            })

        # Latest vs previous.
        spearman_prev = pairwise[-1]["spearman"]

        # Latest vs first.
        first = self._history[0]
        latest = self._history[-1]
        spearman_first = self._spearman_rank_correlation(
            first["top_features"], first["top_importances"],
            latest["top_features"], latest["top_importances"],
        )

        mean_spearman = float(np.mean([p["spearman"] for p in pairwise]))

        # Alert logic.
        alert = spearman_prev < self.shift_threshold
        if alert:
            alert_msg = (
                f"ALERT: Feature importance shift detected between cycles "
                f"'{self._history[-2]['cycle_id']}' and '{latest['cycle_id']}'. "
                f"Spearman rank correlation = {spearman_prev:.4f} "
                f"(threshold = {self.shift_threshold:.2f}). "
                f"Investigate whether the data distribution or feature "
                f"engineering has changed."
            )
        else:
            alert_msg = (
                f"Feature importance is stable. Spearman vs previous cycle = "
                f"{spearman_prev:.4f} (threshold = {self.shift_threshold:.2f})."
            )

        return StabilityReport(
            n_cycles=n,
            latest_cycle_id=latest["cycle_id"],
            spearman_vs_previous=round(spearman_prev, 4),
            spearman_vs_first=round(spearman_first, 4),
            mean_spearman=round(mean_spearman, 4),
            alert=alert,
            alert_message=alert_msg,
            pairwise_correlations=pairwise,
            top_features_latest=latest["top_features"],
        )
