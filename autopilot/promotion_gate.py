"""
Promotion gate for deciding whether a discovered strategy is deployable.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from ..backtest.engine import BacktestResult
from ..config import (
    PROMOTION_MIN_TRADES,
    PROMOTION_MIN_WIN_RATE,
    PROMOTION_MIN_SHARPE,
    PROMOTION_MIN_PROFIT_FACTOR,
    PROMOTION_MAX_DRAWDOWN,
    PROMOTION_MIN_ANNUAL_RETURN,
    PROMOTION_REQUIRE_ADVANCED_CONTRACT,
    PROMOTION_MAX_DSR_PVALUE,
    PROMOTION_MAX_PBO,
    PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED,
    PROMOTION_MAX_CAPACITY_UTILIZATION,
    PROMOTION_MIN_WF_OOS_CORR,
    PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION,
    PROMOTION_MAX_WF_IS_OOS_GAP,
    PROMOTION_MIN_REGIME_POSITIVE_FRACTION,
    PROMOTION_EVENT_MAX_WORST_EVENT_LOSS,
    PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE,
    PROMOTION_EVENT_MIN_REGIME_STABILITY,
    PROMOTION_REQUIRE_STATISTICAL_TESTS,
    PROMOTION_REQUIRE_CPCV,
    PROMOTION_REQUIRE_SPA,
    FOLD_CONSISTENCY_PENALTY_WEIGHT,
    PROMOTION_MAX_STRESS_DRAWDOWN,
    PROMOTION_MIN_STRESS_SHARPE,
    PROMOTION_MAX_TRANSITION_DRAWDOWN,
    PROMOTION_STRESS_REGIMES,
)
from .strategy_discovery import StrategyCandidate


@dataclass
class PromotionDecision:
    """Serializable promotion-gate decision for a single strategy candidate evaluation."""
    candidate: StrategyCandidate
    passed: bool
    score: float
    reasons: List[str]
    metrics: Dict[str, object]

    def to_dict(self) -> Dict:
        """Serialize PromotionDecision to a dictionary."""
        out = asdict(self)
        out["candidate"] = self.candidate.to_dict()
        return out


class PromotionGate:
    """
    Applies hard risk/quality constraints before a strategy can be paper-deployed.
    """

    def __init__(
        self,
        min_trades: int = PROMOTION_MIN_TRADES,
        min_win_rate: float = PROMOTION_MIN_WIN_RATE,
        min_sharpe: float = PROMOTION_MIN_SHARPE,
        min_profit_factor: float = PROMOTION_MIN_PROFIT_FACTOR,
        max_drawdown: float = PROMOTION_MAX_DRAWDOWN,
        min_annual_return: float = PROMOTION_MIN_ANNUAL_RETURN,
        require_advanced_contract: bool = PROMOTION_REQUIRE_ADVANCED_CONTRACT,
        max_dsr_pvalue: float = PROMOTION_MAX_DSR_PVALUE,
        max_pbo: float = PROMOTION_MAX_PBO,
        require_capacity_unconstrained: bool = PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED,
        max_capacity_utilization: float = PROMOTION_MAX_CAPACITY_UTILIZATION,
        min_wf_oos_corr: float = PROMOTION_MIN_WF_OOS_CORR,
        min_wf_positive_fold_fraction: float = PROMOTION_MIN_WF_POSITIVE_FOLD_FRACTION,
        max_wf_is_oos_gap: float = PROMOTION_MAX_WF_IS_OOS_GAP,
        min_regime_positive_fraction: float = PROMOTION_MIN_REGIME_POSITIVE_FRACTION,
        event_max_worst_event_loss: float = PROMOTION_EVENT_MAX_WORST_EVENT_LOSS,
        event_min_surprise_hit_rate: float = PROMOTION_EVENT_MIN_SURPRISE_HIT_RATE,
        event_min_regime_stability: float = PROMOTION_EVENT_MIN_REGIME_STABILITY,
        require_statistical_tests: bool = PROMOTION_REQUIRE_STATISTICAL_TESTS,
        require_cpcv: bool = PROMOTION_REQUIRE_CPCV,
        require_spa: bool = PROMOTION_REQUIRE_SPA,
        weight_fold_consistency: float = FOLD_CONSISTENCY_PENALTY_WEIGHT,
        max_stress_drawdown: float = PROMOTION_MAX_STRESS_DRAWDOWN,
        min_stress_sharpe: float = PROMOTION_MIN_STRESS_SHARPE,
        max_transition_drawdown: float = PROMOTION_MAX_TRANSITION_DRAWDOWN,
        stress_regimes: Optional[List[int]] = None,
    ):
        """Initialize PromotionGate."""
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown
        self.min_annual_return = min_annual_return
        self.require_advanced_contract = require_advanced_contract
        self.max_dsr_pvalue = max_dsr_pvalue
        self.max_pbo = max_pbo
        self.require_capacity_unconstrained = require_capacity_unconstrained
        self.max_capacity_utilization = max_capacity_utilization
        self.min_wf_oos_corr = min_wf_oos_corr
        self.min_wf_positive_fold_fraction = min_wf_positive_fold_fraction
        self.max_wf_is_oos_gap = max_wf_is_oos_gap
        self.min_regime_positive_fraction = min_regime_positive_fraction
        self.event_max_worst_event_loss = event_max_worst_event_loss
        self.event_min_surprise_hit_rate = event_min_surprise_hit_rate
        self.event_min_regime_stability = event_min_regime_stability
        self.require_statistical_tests = require_statistical_tests
        self.require_cpcv = require_cpcv
        self.require_spa = require_spa
        self.weight_fold_consistency = weight_fold_consistency
        self.max_stress_drawdown = max_stress_drawdown
        self.min_stress_sharpe = min_stress_sharpe
        self.max_transition_drawdown = max_transition_drawdown
        self.stress_regimes = stress_regimes if stress_regimes is not None else list(PROMOTION_STRESS_REGIMES)

    def evaluate(
        self,
        candidate: StrategyCandidate,
        result: BacktestResult,
        contract_metrics: Optional[Dict[str, object]] = None,
        event_mode: bool = False,
    ) -> PromotionDecision:
        """Evaluate the requested inputs."""
        metrics = {
            "total_trades": float(result.total_trades),
            "win_rate": float(result.win_rate),
            "sharpe": float(result.sharpe_ratio),
            "profit_factor": float(result.profit_factor),
            "max_drawdown": float(result.max_drawdown),
            "annualized_return": float(result.annualized_return),
            "trades_per_year": float(result.trades_per_year),
        }
        if contract_metrics:
            metrics.update(contract_metrics)

        reasons: List[str] = []
        if result.total_trades < self.min_trades:
            reasons.append(f"insufficient_trades<{self.min_trades}")
        if result.win_rate < self.min_win_rate:
            reasons.append(f"win_rate<{self.min_win_rate:.2f}")
        if result.sharpe_ratio < self.min_sharpe:
            reasons.append(f"sharpe<{self.min_sharpe:.2f}")
        if result.profit_factor < self.min_profit_factor:
            reasons.append(f"profit_factor<{self.min_profit_factor:.2f}")
        if result.max_drawdown < self.max_drawdown:
            reasons.append(f"max_drawdown<{self.max_drawdown:.2f}")
        if result.annualized_return < self.min_annual_return:
            reasons.append(f"annualized_return<{self.min_annual_return:.2f}")

        # Mandatory: reject negative-Sharpe strategies outright.
        # A strategy with negative expected return has no edge and must not
        # be promoted regardless of other metrics.
        if result.sharpe_ratio <= 0:
            reasons.append(f"negative_sharpe({result.sharpe_ratio:.3f})")

        if self.require_advanced_contract:
            dsr_significant = bool(metrics.get("dsr_significant", False))
            dsr_p = float(metrics.get("dsr_p_value", 1.0))
            pbo = metrics.get("pbo", None)
            mc_significant = bool(metrics.get("mc_significant", False))
            mc_p = float(metrics.get("mc_p_value", 1.0))
            capacity_constrained = bool(metrics.get("capacity_constrained", True))
            cap_util = float(metrics.get("capacity_utilization", np.inf))
            wf_oos = float(metrics.get("wf_oos_corr", -np.inf))
            wf_fold_frac = float(metrics.get("wf_positive_fold_fraction", 0.0))
            wf_gap = float(metrics.get("wf_is_oos_gap", np.inf))
            regime_pos = float(metrics.get("regime_positive_fraction", 0.0))

            if (not dsr_significant) or dsr_p > self.max_dsr_pvalue:
                reasons.append(f"dsr_not_significant(p={dsr_p:.4f})")

            # Monte Carlo validation: strategy must beat shuffled null
            if not mc_significant:
                reasons.append(f"mc_not_significant(p={mc_p:.4f})")

            if pbo is None:
                if not event_mode:
                    reasons.append("pbo_unavailable")
            elif float(pbo) > self.max_pbo:
                reasons.append(f"pbo>{self.max_pbo:.2f}")

            if self.require_capacity_unconstrained and capacity_constrained:
                reasons.append("capacity_constrained")
            if cap_util > self.max_capacity_utilization:
                reasons.append(
                    f"capacity_utilization>{self.max_capacity_utilization:.2f}",
                )

            if wf_oos < self.min_wf_oos_corr:
                reasons.append(f"wf_oos_corr<{self.min_wf_oos_corr:.3f}")
            if wf_fold_frac < self.min_wf_positive_fold_fraction:
                reasons.append(
                    f"wf_positive_fold_fraction<{self.min_wf_positive_fold_fraction:.2f}",
                )
            if wf_gap > self.max_wf_is_oos_gap:
                reasons.append(f"wf_is_oos_gap>{self.max_wf_is_oos_gap:.2f}")
            if (not event_mode) and regime_pos < self.min_regime_positive_fraction:
                reasons.append(
                    f"regime_positive_fraction<{self.min_regime_positive_fraction:.2f}",
                )

            # ── New validation gates ──
            if self.require_statistical_tests:
                stat_tests_pass = bool(metrics.get("stat_tests_pass", False))
                if not stat_tests_pass:
                    reasons.append("stat_tests_failed")

            if self.require_cpcv:
                cpcv_passes = bool(metrics.get("cpcv_passes", False))
                if not cpcv_passes:
                    reasons.append("cpcv_failed")

            if self.require_spa:
                spa_passes = bool(metrics.get("spa_passes", False))
                if not spa_passes:
                    spa_p = float(metrics.get("spa_pvalue", 1.0))
                    reasons.append(f"spa_failed(p={spa_p:.4f})")

            # ── Stress-regime bucket gates (SPEC-V02) ──
            # Reject strategies with catastrophic losses during shock periods
            # even if overall metrics look acceptable.
            regime_perf = result.regime_performance
            if regime_perf:
                for regime_code in self.stress_regimes:
                    rp = regime_perf.get(regime_code)
                    if rp is None:
                        continue
                    # Only gate when there are enough trades for meaningful statistics
                    if rp.get("n_trades", 0) < 5:
                        continue
                    regime_dd = rp.get("max_drawdown", 0.0)
                    regime_sharpe = rp.get("sharpe", 0.0)
                    # max_drawdown values are negative; compare magnitude
                    if regime_dd < -self.max_stress_drawdown:
                        reasons.append(
                            f"stress_regime_{regime_code}_drawdown"
                            f"({regime_dd:.1%} exceeds -{self.max_stress_drawdown:.1%})"
                        )
                    if regime_sharpe < self.min_stress_sharpe:
                        reasons.append(
                            f"stress_regime_{regime_code}_sharpe"
                            f"({regime_sharpe:.2f} below {self.min_stress_sharpe:.2f})"
                        )
                # Store stress regime metrics for diagnostics
                for regime_code in self.stress_regimes:
                    rp = regime_perf.get(regime_code)
                    if rp is not None:
                        metrics[f"stress_regime_{regime_code}_drawdown"] = rp.get("max_drawdown", 0.0)
                        metrics[f"stress_regime_{regime_code}_sharpe"] = rp.get("sharpe", 0.0)
                        metrics[f"stress_regime_{regime_code}_n_trades"] = rp.get("n_trades", 0)

        # Event-strategy contract checks (only enforced when metrics are supplied).
        if "worst_event_loss" in metrics:
            worst_event_loss = float(metrics.get("worst_event_loss", -np.inf))
            if worst_event_loss < self.event_max_worst_event_loss:
                reasons.append(
                    f"worst_event_loss<{self.event_max_worst_event_loss:.2f}",
                )
        if "surprise_hit_rate" in metrics:
            surprise_hit = float(metrics.get("surprise_hit_rate", 0.0))
            if surprise_hit < self.event_min_surprise_hit_rate:
                reasons.append(
                    f"surprise_hit_rate<{self.event_min_surprise_hit_rate:.2f}",
                )
        if "event_regime_stability" in metrics:
            stability = float(metrics.get("event_regime_stability", 0.0))
            if stability < self.event_min_regime_stability:
                reasons.append(
                    f"event_regime_stability<{self.event_min_regime_stability:.2f}",
                )

        # Composite score used for ranking passers (not pass/fail itself).
        score = (
            1.30 * result.sharpe_ratio
            + 0.80 * result.annualized_return
            + 0.35 * result.win_rate
            + 0.20 * min(result.profit_factor, 5.0)
            + 0.01 * min(result.total_trades, 5000)
            + 0.80 * max(result.max_drawdown, -0.50)
        )
        if "wf_oos_corr" in metrics:
            score += 0.75 * float(metrics.get("wf_oos_corr", 0.0))
        if "regime_positive_fraction" in metrics:
            score += 0.40 * float(metrics.get("regime_positive_fraction", 0.0))
        if "dsr_significant" in metrics and bool(metrics.get("dsr_significant", False)):
            score += 0.50
        if bool(metrics.get("mc_significant", False)):
            score += 0.45
        if bool(metrics.get("stat_tests_pass", False)):
            score += 0.30
        if bool(metrics.get("cpcv_passes", False)):
            score += 0.35
        if bool(metrics.get("spa_passes", False)):
            score += 0.40

        # Stress resilience bonus (SPEC-V02): strategies that survive
        # stress regimes with positive Sharpe receive a ranking bonus.
        stress_sharpes = []
        regime_perf = result.regime_performance
        if regime_perf:
            for rc in self.stress_regimes:
                rp = regime_perf.get(rc)
                if rp is not None and rp.get("n_trades", 0) >= 5:
                    stress_sharpes.append(rp.get("sharpe", 0.0))
        if stress_sharpes:
            avg_stress_sharpe = float(np.mean(stress_sharpes))
            # Clamp contribution: reward up to +0.50, penalize down to -0.30
            stress_bonus = float(np.clip(0.25 * avg_stress_sharpe, -0.30, 0.50))
            score += stress_bonus
            metrics["avg_stress_sharpe"] = avg_stress_sharpe
            metrics["stress_resilience_bonus"] = stress_bonus

        # Fold consistency reward (Spec 04): strategies with stable
        # performance across walk-forward folds receive a score bonus.
        fold_metrics = metrics.get("fold_metrics")
        if fold_metrics and len(fold_metrics) >= 3:
            fold_consistency = self._compute_fold_consistency(fold_metrics)
            score += self.weight_fold_consistency * fold_consistency
            metrics["fold_consistency"] = fold_consistency

        return PromotionDecision(
            candidate=candidate,
            passed=len(reasons) == 0,
            score=float(score),
            reasons=reasons,
            metrics=metrics,
        )

    @staticmethod
    def _compute_fold_consistency(fold_metrics: List[Dict]) -> float:
        """Compute fold consistency score from per-fold Sharpe ratios.

        Measures how stable a strategy's performance is across walk-forward
        folds.  High consistency (close to 1.0) means the strategy works
        across different time periods; low consistency (close to 0.0) means
        the strategy's edge is regime-dependent or unreliable.

        Formula: ``clip(1 - std(sharpes) / |mean(sharpes)|, 0, 1)``

        Args:
            fold_metrics: List of dicts, each containing at least a
                ``sharpe_estimate`` (or ``sharpe``) key.

        Returns:
            Float in [0, 1].  Returns 0.0 if mean Sharpe <= 0.
        """
        sharpes = []
        for fm in fold_metrics:
            s = fm.get("sharpe_estimate", fm.get("sharpe", None))
            if s is not None and np.isfinite(s):
                sharpes.append(float(s))

        if len(sharpes) < 2:
            return 1.0  # insufficient data to penalize

        mean_sharpe = float(np.mean(sharpes))
        if mean_sharpe <= 0:
            return 0.0

        std_sharpe = float(np.std(sharpes))
        consistency = 1.0 - (std_sharpe / abs(mean_sharpe))
        return float(np.clip(consistency, 0.0, 1.0))

    def evaluate_event_strategy(
        self,
        candidate: StrategyCandidate,
        result: BacktestResult,
        event_metrics: Dict[str, object],
        contract_metrics: Optional[Dict[str, object]] = None,
    ) -> PromotionDecision:
        """
        Evaluate an event strategy using standard + event-specific contract checks.
        """
        merged = dict(contract_metrics or {})
        merged.update(event_metrics or {})
        return self.evaluate(
            candidate=candidate,
            result=result,
            contract_metrics=merged,
            event_mode=True,
        )

    @staticmethod
    def rank(decisions: List[PromotionDecision]) -> List[PromotionDecision]:
        """Rank candidate inputs."""
        return sorted(decisions, key=lambda d: d.score, reverse=True)
