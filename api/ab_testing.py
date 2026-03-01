"""
A/B testing framework for strategy evaluation.

Enables controlled comparison of strategy variants (e.g., different Kelly
fractions, regime detectors, feature sets) by splitting paper trading
allocation between a control and treatment group.

Statistical methods account for time-series autocorrelation (block bootstrap,
Newey-West HAC).  Ticker-level assignment prevents contamination.  Sequential
testing with O'Brien-Fleming alpha spending supports early stopping.

Constraints
-----------
- Paper trading only (no live capital)
- Maximum 3 concurrent active tests
- No auto-promotion of winners (human review required)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Sanitize a name for safe use in filesystem paths.

    Strips path separators, parent-directory traversals, and null bytes.
    Only alphanumeric characters, hyphens, and underscores are retained.

    Raises
    ------
    ValueError
        If the sanitized result is empty.
    """
    # Remove null bytes
    cleaned = name.replace("\x00", "")
    # Keep only alphanumeric, hyphens, underscores
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", cleaned)
    if not cleaned:
        raise ValueError(
            f"Invalid name {name!r}: must contain at least one "
            "alphanumeric character, hyphen, or underscore."
        )
    return cleaned

# Config overrides that A/B tests are allowed to modify.
# max_positions is excluded to prevent imbalanced capital allocation.
ALLOWED_OVERRIDES = frozenset({
    "entry_threshold",
    "confidence_threshold",
    "position_size_pct",
    "max_holding_days",
    "kelly_fraction",
    "use_risk_management",
})

MAX_CONCURRENT_TESTS = 3


@dataclass
class ABVariant:
    """One arm of an A/B test."""

    name: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    allocation: float = 0.5
    trades: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def returns(self) -> np.ndarray:
        """Return array of net returns from recorded trades."""
        if not self.trades:
            return np.array([], dtype=float)
        return np.array(
            [float(t.get("net_return", 0.0)) for t in self.trades],
            dtype=float,
        )

    @property
    def mean_return(self) -> float:
        rets = self.returns()
        if len(rets) == 0:
            return 0.0
        return float(np.mean(rets))

    @property
    def sharpe(self) -> float:
        rets = self.returns()
        if len(rets) < 2:
            return 0.0
        std = float(rets.std())
        if std < 1e-10:
            return 0.0
        return float(rets.mean() / std * np.sqrt(252))

    @property
    def sortino(self) -> float:
        rets = self.returns()
        if len(rets) < 2:
            return 0.0
        downside = rets[rets < 0]
        if len(downside) == 0:
            return 0.0
        down_std = float(downside.std())
        if down_std < 1e-10:
            return 0.0
        return float(rets.mean() / down_std * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        rets = self.returns()
        if len(rets) == 0:
            return 0.0
        cumulative = np.cumprod(1.0 + rets)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns))

    @property
    def win_rate(self) -> float:
        rets = self.returns()
        if len(rets) == 0:
            return 0.0
        return float(np.mean(rets > 0))

    @property
    def profit_factor(self) -> float:
        rets = self.returns()
        gains = rets[rets > 0]
        losses = rets[rets < 0]
        total_loss = float(np.abs(losses).sum()) if len(losses) > 0 else 0.0
        total_gain = float(gains.sum()) if len(gains) > 0 else 0.0
        if total_loss < 1e-10:
            return float("inf") if total_gain > 0 else 0.0
        return total_gain / total_loss

    @property
    def turnover(self) -> float:
        """Average number of trades per day, based on date span."""
        if len(self.trades) < 2:
            return 0.0
        dates = sorted(set(
            t.get("entry_date", t.get("exit_date", ""))
            for t in self.trades
            if t.get("entry_date") or t.get("exit_date")
        ))
        if len(dates) < 2:
            return 0.0
        try:
            from datetime import date as dt_date
            first = dt_date.fromisoformat(str(dates[0]))
            last = dt_date.fromisoformat(str(dates[-1]))
            span = (last - first).days
            if span <= 0:
                return 0.0
            return len(self.trades) / span
        except (ValueError, TypeError):
            return 0.0

    @property
    def total_transaction_costs(self) -> float:
        return float(sum(
            float(t.get("transaction_cost", 0.0)) for t in self.trades
        ))


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
    min_trades: int = 50
    confidence_level: float = 0.95
    _ticker_assignments: Dict[str, str] = field(
        default_factory=dict, repr=False,
    )

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ── Trade recording ────────────────────────────────────────────────

    def record_trade(self, variant: str, trade: Dict[str, Any]) -> None:
        """Record a trade result for a variant."""
        target = self._resolve_variant(variant)
        target.trades.append(trade)

    def _resolve_variant(self, variant: str) -> ABVariant:
        """Resolve a variant name to the corresponding ABVariant object."""
        if variant == self.control.name:
            return self.control
        if variant == self.treatment.name:
            return self.treatment
        # Legacy compat: "control"/"treatment" literals
        if variant == "control":
            return self.control
        if variant == "treatment":
            return self.treatment
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Use {self.control.name!r} or {self.treatment.name!r}."
        )

    # ── T2: Ticker-level deterministic assignment ──────────────────────

    def assign_variant(self, ticker: str) -> str:
        """Assign ticker to variant deterministically based on hash.

        Same ticker always maps to the same variant for this test,
        preventing contamination (same ticker in both arms).
        """
        if ticker in self._ticker_assignments:
            return self._ticker_assignments[ticker]

        # Deterministic hash: same test_id + ticker → same assignment
        hash_input = f"{self.test_id}:{ticker}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Assign based on allocation ratio (hash is uniformly distributed)
        if (hash_val % 1000) / 1000.0 < self.control.allocation:
            variant = self.control.name
        else:
            variant = self.treatment.name

        self._ticker_assignments[ticker] = variant
        return variant

    def get_variant_config(self, variant: str) -> Dict[str, Any]:
        """Return the config overrides for a given variant name."""
        target = self._resolve_variant(variant)
        return dict(target.config_overrides)

    # ── T1: Block bootstrap + Newey-West statistical testing ───────────

    def _block_resample(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Circular block bootstrap resampling."""
        n = len(data)
        if n == 0:
            return data
        n_blocks = int(np.ceil(n / block_size))
        indices: List[int] = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n)
            indices.extend(range(start, start + block_size))
        indices = [i % n for i in indices[:n]]
        return data[np.array(indices)]

    def _block_bootstrap_test(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        n_bootstrap: int = 1000,
        block_size: int = 10,
    ) -> float:
        """Block bootstrap test for difference in means.

        Respects time-series autocorrelation by resampling in blocks
        rather than individual observations.

        Returns
        -------
        p_value : float
            Two-sided p-value under the null of equal means.
        """
        observed_diff = returns_a.mean() - returns_b.mean()

        # Pool all returns under null hypothesis
        pooled = np.concatenate([returns_a, returns_b])
        n_a = len(returns_a)

        bootstrap_diffs = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            boot_pooled = self._block_resample(pooled, block_size)
            boot_a = boot_pooled[:n_a]
            boot_b = boot_pooled[n_a:n_a + len(returns_b)]
            bootstrap_diffs[i] = boot_a.mean() - boot_b.mean()

        p_value = float(np.mean(np.abs(bootstrap_diffs) >= abs(observed_diff)))
        return p_value

    @staticmethod
    def _newey_west_variance(centered: np.ndarray, max_lag: int) -> float:
        """Compute Newey-West HAC variance estimate for a single series.

        Parameters
        ----------
        centered : np.ndarray
            Mean-centered return series.
        max_lag : int
            Maximum lag for Bartlett kernel.

        Returns
        -------
        float
            HAC-consistent variance estimate (non-negative).
        """
        T = len(centered)
        gamma_0 = float(np.var(centered, ddof=0))
        nw_var = gamma_0

        for lag in range(1, min(max_lag + 1, T)):
            weight = 1.0 - lag / (max_lag + 1)  # Bartlett kernel
            gamma_lag = float(np.mean(centered[lag:] * centered[:-lag]))
            nw_var += 2.0 * weight * gamma_lag

        return max(nw_var, 1e-20)

    def _newey_west_test(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        max_lag: int = 10,
    ) -> Tuple[float, float]:
        """HAC-consistent t-test using Newey-West standard errors.

        Computes Newey-West variance separately for each arm, then
        combines them for the two-sample t-test.

        Returns
        -------
        t_stat : float
        p_value : float
        """
        diff = returns_a.mean() - returns_b.mean()

        nw_var_a = self._newey_west_variance(
            returns_a - returns_a.mean(), max_lag
        )
        nw_var_b = self._newey_west_variance(
            returns_b - returns_b.mean(), max_lag
        )

        se = np.sqrt(nw_var_a / len(returns_a) + nw_var_b / len(returns_b))
        if se < 1e-20:
            return 0.0, 1.0
        t_stat = diff / se
        p_value = float(2.0 * (1.0 - norm.cdf(abs(t_stat))))
        return float(t_stat), p_value

    # ── T3: Power analysis ─────────────────────────────────────────────

    def compute_required_samples(
        self,
        min_detectable_effect: float = 0.005,
        power: float = 0.80,
        alpha: float = 0.05,
        return_std: float = 0.02,
    ) -> int:
        """Compute minimum trades per variant for desired statistical power.

        Parameters
        ----------
        min_detectable_effect : float
            Minimum return difference to detect (default 50 bps).
        power : float
            Desired statistical power (default 0.80).
        alpha : float
            Significance level (default 0.05, two-sided).
        return_std : float
            Expected standard deviation of trade returns (default 2%).

        Returns
        -------
        int
            Minimum trades per variant, adjusted for autocorrelation.
        """
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)

        # Two-sample test: n = 2 * (z_alpha + z_beta)^2 * sigma^2 / delta^2
        if min_detectable_effect <= 0:
            return 50
        n = 2.0 * (z_alpha + z_beta) ** 2 * return_std ** 2 / min_detectable_effect ** 2

        # Adjust for autocorrelation (assume AR(1) with rho ~ 0.10)
        rho = 0.10
        autocorr_factor = (1.0 + rho) / (1.0 - rho)
        n_adjusted = int(np.ceil(n * autocorr_factor))

        return max(n_adjusted, 50)

    # ── T4: Sequential testing / early stopping ────────────────────────

    def check_early_stopping(self, interim_looks: int = 5) -> Dict[str, Any]:
        """Check if test can be stopped early using O'Brien-Fleming alpha spending.

        Parameters
        ----------
        interim_looks : int
            Planned number of interim analyses (default 5).

        Returns
        -------
        dict
            Contains 'stop' (bool), 'reason', 'info_fraction', 'p_value',
            'alpha_boundary', and 'trades_remaining'.
        """
        n_control = self.control.n_trades
        n_treatment = self.treatment.n_trades

        if n_control < 20 or n_treatment < 20:
            return {
                "stop": False,
                "reason": "Insufficient data for interim analysis",
                "info_fraction": 0.0,
                "p_value": None,
                "alpha_boundary": None,
                "trades_remaining": None,
            }

        required = self.compute_required_samples()

        # Information fraction: how much data do we have vs required?
        info_fraction = min(1.0, min(n_control, n_treatment) / required)

        # O'Brien-Fleming alpha spending: very conservative early, liberal late
        # alpha_spent(t) = 2 * (1 - Phi(z_alpha/2 / sqrt(t)))
        alpha = 1.0 - self.confidence_level
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        if info_fraction > 0:
            z_boundary = z_alpha / np.sqrt(info_fraction)
        else:
            z_boundary = float("inf")
        spent_alpha = float(2.0 * (1.0 - norm.cdf(z_boundary)))

        # Compute current test statistic using HAC-consistent test
        ctrl_rets = self.control.returns()
        treat_rets = self.treatment.returns()
        _, p_value = self._newey_west_test(ctrl_rets, treat_rets)

        can_stop = p_value < spent_alpha

        # Safety check: stop if treatment is significantly WORSE
        treatment_worse = (
            self.treatment.mean_return < self.control.mean_return
            and p_value < 0.01
        )

        if can_stop:
            reason = "Treatment significantly better"
        elif treatment_worse:
            reason = "Treatment significantly worse — recommend stopping"
        else:
            reason = "Not yet significant"

        return {
            "stop": can_stop or treatment_worse,
            "reason": reason,
            "info_fraction": float(info_fraction),
            "p_value": float(p_value),
            "alpha_boundary": float(spent_alpha),
            "trades_remaining": max(
                0, required - min(n_control, n_treatment)
            ),
        }

    # ── Results & reporting ────────────────────────────────────────────

    def get_results(self) -> Dict[str, Any]:
        """Compute A/B test results with time-series-aware significance.

        Uses block bootstrap as primary test, Newey-West as secondary.
        Reports comprehensive metrics for both variants.
        """
        ctrl_rets = self.control.returns()
        treat_rets = self.treatment.returns()

        result: Dict[str, Any] = {
            "test_id": self.test_id,
            "name": self.name,
            "status": self.status,
            "control": self._variant_metrics(self.control),
            "treatment": self._variant_metrics(self.treatment),
            "sufficient_data": False,
            "significant": False,
            "p_value": None,
            "p_value_bootstrap": None,
            "p_value_newey_west": None,
            "winner": None,
            "required_samples": self.compute_required_samples(),
            "early_stopping": self.check_early_stopping()
            if (len(ctrl_rets) >= 20 and len(treat_rets) >= 20)
            else None,
        }

        if len(ctrl_rets) < self.min_trades or len(treat_rets) < self.min_trades:
            return result

        result["sufficient_data"] = True

        # Block bootstrap (primary — respects autocorrelation)
        p_bootstrap = self._block_bootstrap_test(ctrl_rets, treat_rets)
        result["p_value_bootstrap"] = float(p_bootstrap)

        # Newey-West HAC (secondary)
        t_stat, p_nw = self._newey_west_test(ctrl_rets, treat_rets)
        result["p_value_newey_west"] = float(p_nw)

        # Use the more conservative (larger) p-value
        p_value = max(p_bootstrap, p_nw)
        result["p_value"] = float(p_value)

        alpha = 1.0 - self.confidence_level
        result["significant"] = p_value < alpha

        if result["significant"]:
            result["winner"] = (
                "treatment"
                if self.treatment.mean_return > self.control.mean_return
                else "control"
            )

        return result

    def get_test_report(self) -> Dict[str, Any]:
        """Comprehensive comparison report with per-regime breakdown."""
        results = self.get_results()

        # Per-regime performance
        for variant_key, variant_obj in [
            ("control", self.control),
            ("treatment", self.treatment),
        ]:
            regime_breakdown: Dict[str, Dict[str, Any]] = {}
            for trade in variant_obj.trades:
                regime = str(trade.get("entry_regime", "unknown"))
                if regime not in regime_breakdown:
                    regime_breakdown[regime] = {"returns": [], "count": 0}
                regime_breakdown[regime]["returns"].append(
                    float(trade.get("net_return", 0.0))
                )
                regime_breakdown[regime]["count"] += 1

            for regime, data in regime_breakdown.items():
                rets = np.array(data["returns"])
                std = float(rets.std()) if len(rets) > 1 else 0.0
                data["mean_return"] = float(rets.mean())
                data["sharpe"] = (
                    float(rets.mean() / std * np.sqrt(252))
                    if std > 1e-10 else 0.0
                )
                del data["returns"]

            results[variant_key]["regime_breakdown"] = regime_breakdown

        results["ticker_assignments"] = dict(self._ticker_assignments)
        return results

    @staticmethod
    def _variant_metrics(variant: ABVariant) -> Dict[str, Any]:
        """Compute comprehensive metrics for a single variant."""
        return {
            "name": variant.name,
            "n_trades": variant.n_trades,
            "mean_return": variant.mean_return,
            "sharpe": variant.sharpe,
            "sortino": variant.sortino,
            "max_drawdown": variant.max_drawdown,
            "win_rate": variant.win_rate,
            "profit_factor": variant.profit_factor,
            "turnover": variant.turnover,
            "total_transaction_costs": variant.total_transaction_costs,
        }

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (excludes raw trades for compactness)."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "control": {
                "name": self.control.name,
                "config": self.control.config_overrides,
                "allocation": self.control.allocation,
                "n_trades": self.control.n_trades,
            },
            "treatment": {
                "name": self.treatment.name,
                "config": self.treatment.config_overrides,
                "allocation": self.treatment.allocation,
                "n_trades": self.treatment.n_trades,
            },
            "created_at": self.created_at,
            "status": self.status,
            "min_trades": self.min_trades,
            "confidence_level": self.confidence_level,
            "ticker_assignments": dict(self._ticker_assignments),
        }


class ABTestRegistry:
    """Manages A/B tests with JSON + parquet persistence.

    Metadata is stored as JSON; full trade history per variant is stored
    as parquet for efficient storage and reproducibility.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(
            storage_path or Path("data/ab_tests.json")
        )
        self._tests: Dict[str, ABTest] = {}
        self._load()

    # ── CRUD ───────────────────────────────────────────────────────────

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
        """Create a new A/B test.

        Raises
        ------
        ValueError
            If MAX_CONCURRENT_TESTS active tests already exist.
        """
        active_count = sum(
            1 for t in self._tests.values() if t.status == "active"
        )
        if active_count >= MAX_CONCURRENT_TESTS:
            raise ValueError(
                f"Cannot create test: {active_count} active tests already "
                f"exist (max {MAX_CONCURRENT_TESTS}). Complete or cancel "
                f"an existing test first."
            )

        # Sanitize variant names to prevent path traversal
        control_name = _sanitize_name(control_name)
        treatment_name = _sanitize_name(treatment_name)

        # Validate overrides
        for key in (control_overrides or {}):
            if key not in ALLOWED_OVERRIDES:
                raise ValueError(
                    f"Config override {key!r} not allowed. "
                    f"Allowed: {sorted(ALLOWED_OVERRIDES)}"
                )
        for key in (treatment_overrides or {}):
            if key not in ALLOWED_OVERRIDES:
                raise ValueError(
                    f"Config override {key!r} not allowed. "
                    f"Allowed: {sorted(ALLOWED_OVERRIDES)}"
                )

        test = ABTest(
            test_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            control=ABVariant(
                name=control_name,
                config_overrides=control_overrides or {},
                allocation=allocation,
            ),
            treatment=ABVariant(
                name=treatment_name,
                config_overrides=treatment_overrides or {},
                allocation=1.0 - allocation,
            ),
            min_trades=min_trades,
        )
        self._tests[test.test_id] = test
        self._save()
        logger.info(
            "Created A/B test %s: %s vs %s",
            test.test_id, control_name, treatment_name,
        )
        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        return self._tests.get(test_id)

    def get_active_test(self) -> Optional[ABTest]:
        """Return the first active A/B test (for paper trader integration)."""
        for test in self._tests.values():
            if test.status == "active":
                return test
        return None

    def list_tests(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        tests = list(self._tests.values())
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

    def cancel_test(self, test_id: str) -> bool:
        """Cancel an active test."""
        test = self._tests.get(test_id)
        if not test or test.status != "active":
            return False
        test.status = "cancelled"
        self._save()
        logger.info("Cancelled A/B test %s", test_id)
        return True

    # ── T5: Full persistence (metadata JSON + trades parquet) ──────────

    def _trades_dir(self) -> Path:
        return self.storage_path.parent / "ab_trades"

    def _save(self) -> None:
        """Persist metadata as JSON and trade history as parquet."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {tid: t.to_dict() for tid, t in self._tests.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            logger.warning("Failed to save A/B test metadata: %s", e)

        # Persist trade history as parquet per variant
        try:
            import pandas as pd

            trades_dir = self._trades_dir()
            trades_dir.mkdir(parents=True, exist_ok=True)

            for test in self._tests.values():
                safe_test_id = _sanitize_name(test.test_id)
                for variant in [test.control, test.treatment]:
                    safe_name = _sanitize_name(variant.name)
                    parquet_path = (
                        trades_dir / f"{safe_test_id}_{safe_name}.parquet"
                    )
                    if variant.trades:
                        df = pd.DataFrame(variant.trades)
                        df.to_parquet(parquet_path, index=False)
                    elif parquet_path.exists():
                        # No trades but file exists — keep it
                        pass
        except ImportError:
            logger.warning(
                "pandas not available; trade history not persisted to parquet"
            )
        except OSError as e:
            logger.warning("Failed to save trade history: %s", e)

    def _load(self) -> None:
        """Load metadata from JSON and trade history from parquet."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load A/B test metadata: %s", e)
            return

        for tid, meta in data.items():
            control_meta = meta.get("control", {})
            treatment_meta = meta.get("treatment", {})
            test = ABTest(
                test_id=tid,
                name=meta.get("name", ""),
                description=meta.get("description", ""),
                control=ABVariant(
                    name=control_meta.get("name", "control"),
                    config_overrides=control_meta.get("config", {}),
                    allocation=control_meta.get("allocation", 0.5),
                ),
                treatment=ABVariant(
                    name=treatment_meta.get("name", "treatment"),
                    config_overrides=treatment_meta.get("config", {}),
                    allocation=treatment_meta.get("allocation", 0.5),
                ),
                created_at=meta.get("created_at", ""),
                status=meta.get("status", "active"),
                min_trades=meta.get("min_trades", 50),
                confidence_level=meta.get("confidence_level", 0.95),
            )

            # Restore ticker assignments
            test._ticker_assignments = dict(
                meta.get("ticker_assignments", {})
            )

            # Restore trades from parquet
            self._load_trades(test)
            self._tests[tid] = test

    def _load_trades(self, test: ABTest) -> None:
        """Load trade history from parquet files into variant objects."""
        try:
            import pandas as pd
        except ImportError:
            return

        trades_dir = self._trades_dir()
        if not trades_dir.exists():
            return

        for variant in [test.control, test.treatment]:
            safe_test_id = _sanitize_name(test.test_id)
            safe_name = _sanitize_name(variant.name)
            parquet_path = (
                trades_dir / f"{safe_test_id}_{safe_name}.parquet"
            )
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                    variant.trades = df.to_dict("records")
                except Exception as e:
                    logger.warning(
                        "Failed to load trades for %s/%s: %s",
                        test.test_id, variant.name, e,
                    )
