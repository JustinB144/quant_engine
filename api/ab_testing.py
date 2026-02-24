"""
A/B testing framework for strategy evaluation.

Enables controlled comparison of strategy variants (e.g., different Kelly
fractions, regime detectors, feature sets) by splitting capital or paper
trading allocation between a control and treatment group.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ABVariant:
    """One arm of an A/B test."""

    name: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    allocation: float = 0.5  # fraction of capital/trades allocated
    trades: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def mean_return(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean([t.get("net_return", 0.0) for t in self.trades]))

    @property
    def sharpe(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        rets = np.array([t.get("net_return", 0.0) for t in self.trades])
        std = rets.std()
        if std < 1e-10:
            return 0.0
        return float(rets.mean() / std * np.sqrt(252))


@dataclass
class ABTest:
    """An A/B test comparing two strategy variants."""

    test_id: str
    name: str
    description: str
    control: ABVariant
    treatment: ABVariant
    created_at: str = ""
    status: str = "active"  # active, completed, cancelled
    min_trades: int = 50  # minimum trades before significance test
    confidence_level: float = 0.95

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def record_trade(self, variant: str, trade: Dict[str, Any]) -> None:
        """Record a trade result for a variant."""
        if variant == "control":
            self.control.trades.append(trade)
        elif variant == "treatment":
            self.treatment.trades.append(trade)
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'control' or 'treatment'.")

    def assign_variant(self) -> str:
        """Randomly assign a trade to control or treatment based on allocation."""
        if np.random.random() < self.control.allocation:
            return "control"
        return "treatment"

    def get_results(self) -> Dict[str, Any]:
        """Compute A/B test results with statistical significance."""
        ctrl_rets = np.array([t.get("net_return", 0.0) for t in self.control.trades])
        treat_rets = np.array([t.get("net_return", 0.0) for t in self.treatment.trades])

        result: Dict[str, Any] = {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status,
            "control": {
                "name": self.control.name,
                "n_trades": self.control.n_trades,
                "mean_return": self.control.mean_return,
                "sharpe": self.control.sharpe,
            },
            "treatment": {
                "name": self.treatment.name,
                "n_trades": self.treatment.n_trades,
                "mean_return": self.treatment.mean_return,
                "sharpe": self.treatment.sharpe,
            },
            "sufficient_data": False,
            "significant": False,
            "p_value": None,
            "winner": None,
        }

        # Need minimum trades in both arms
        if len(ctrl_rets) < self.min_trades or len(treat_rets) < self.min_trades:
            return result

        result["sufficient_data"] = True

        # Two-sample t-test (Welch's)
        try:
            from scipy.stats import ttest_ind

            t_stat, p_value = ttest_ind(treat_rets, ctrl_rets, equal_var=False)
            alpha = 1.0 - self.confidence_level
            result["p_value"] = float(p_value)
            result["significant"] = p_value < alpha

            if result["significant"]:
                result["winner"] = "treatment" if t_stat > 0 else "control"
        except ImportError:
            # scipy not available — use simple mean comparison
            result["p_value"] = None
            if abs(self.treatment.mean_return - self.control.mean_return) > 0.001:
                result["winner"] = (
                    "treatment"
                    if self.treatment.mean_return > self.control.mean_return
                    else "control"
                )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (excludes raw trades for compactness)."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "control": {"name": self.control.name, "config": self.control.config_overrides, "n_trades": self.control.n_trades},
            "treatment": {"name": self.treatment.name, "config": self.treatment.config_overrides, "n_trades": self.treatment.n_trades},
            "created_at": self.created_at,
            "status": self.status,
            "min_trades": self.min_trades,
        }


class ABTestRegistry:
    """Manages A/B tests with JSON persistence."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/ab_tests.json")
        self._tests: Dict[str, ABTest] = {}

    def create_test(
        self,
        name: str,
        description: str,
        control_name: str,
        treatment_name: str,
        control_overrides: Optional[Dict[str, Any]] = None,
        treatment_overrides: Optional[Dict[str, Any]] = None,
        allocation: float = 0.5,
        min_trades: int = 50,
    ) -> ABTest:
        """Create a new A/B test."""
        test = ABTest(
            test_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            control=ABVariant(name=control_name, config_overrides=control_overrides or {}, allocation=allocation),
            treatment=ABVariant(name=treatment_name, config_overrides=treatment_overrides or {}, allocation=1.0 - allocation),
            min_trades=min_trades,
        )
        self._tests[test.test_id] = test
        self._save()
        logger.info("Created A/B test %s: %s vs %s", test.test_id, control_name, treatment_name)
        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        return self._tests.get(test_id)

    def list_tests(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        tests = self._tests.values()
        if status:
            tests = [t for t in tests if t.status == status]
        return [t.to_dict() for t in tests]

    def complete_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Mark test as complete and return results."""
        test = self._tests.get(test_id)
        if not test:
            return None
        test.status = "completed"
        self._save()
        return test.get_results()

    def _save(self) -> None:
        """Persist tests to JSON (metadata only, not raw trades)."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {tid: t.to_dict() for tid, t in self._tests.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save A/B test registry: %s", e)
