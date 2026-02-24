"""
System Diagnostics — explains WHY performance is degrading.

Correlates system health metrics with P&L to identify root causes of
underperformance.  Each diagnostic check produces an evidence-based
finding with severity and recommended action.

Diagnostic checks:
    1. ALPHA_DECAY: Model signal quality declining (IC trend negative)
    2. UNFAVORABLE_REGIME: Market in difficult regime for long periods
    3. EXECUTION_DEGRADATION: Slippage/impact costs increasing
    4. STALE_DATA: Data not refreshed recently
    5. MODEL_STALE: Too long since last model retrain
    6. FEATURE_SHIFT: Feature distributions have drifted from training
    7. DRAWDOWN_LOCKOUT: Portfolio in drawdown recovery (sizing reduced)
    8. CONCENTRATION_RISK: Portfolio too concentrated in few names/sectors

Integration points:
    - API: /api/diagnostics endpoint
    - Health dashboard: self-diagnostic panel
    - Autopilot: diagnostic-driven retrain decisions
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticFinding:
    """A single diagnostic finding with evidence and recommendation."""
    cause: str
    severity: str  # "HIGH", "MEDIUM", "LOW"
    evidence: str
    recommendation: str
    metric_value: Optional[float] = None
    metric_threshold: Optional[float] = None


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for the system."""
    status: str  # "PERFORMING", "UNDERPERFORMING", "UNKNOWN"
    diagnostics: List[DiagnosticFinding] = field(default_factory=list)
    primary_cause: str = "UNKNOWN"
    timestamp: str = ""
    recent_return: Optional[float] = None
    recent_sharpe: Optional[float] = None


class SystemDiagnostics:
    """
    Correlate system metrics with performance to identify root causes
    of degradation.

    Usage:
        diag = SystemDiagnostics()
        report = diag.diagnose_performance(
            equity_curve=equity_df,
            health_history=health_dict,
            regime_history=regime_dict,
        )
        if report.status == "UNDERPERFORMING":
            for finding in report.diagnostics:
                print(f"{finding.severity}: {finding.cause} — {finding.evidence}")
    """

    def __init__(
        self,
        underperformance_window: int = 20,
        alpha_decay_threshold: float = 50.0,
        execution_quality_threshold: float = 50.0,
        data_staleness_days: int = 7,
        model_staleness_days: int = 45,
        drawdown_threshold: float = -0.05,
        concentration_threshold: float = 0.30,
    ):
        """
        Args:
            underperformance_window: Days of recent returns to evaluate.
            alpha_decay_threshold: Health score below this = alpha decay.
            execution_quality_threshold: Score below this = execution issue.
            data_staleness_days: Data older than this triggers warning.
            model_staleness_days: Model older than this triggers warning.
            drawdown_threshold: Current drawdown worse than this = locked out.
            concentration_threshold: Top position weight above this = concentrated.
        """
        self.underperformance_window = underperformance_window
        self.alpha_decay_threshold = alpha_decay_threshold
        self.execution_quality_threshold = execution_quality_threshold
        self.data_staleness_days = data_staleness_days
        self.model_staleness_days = model_staleness_days
        self.drawdown_threshold = drawdown_threshold
        self.concentration_threshold = concentration_threshold

    def diagnose_performance(
        self,
        equity_curve: Optional[Dict] = None,
        health_history: Optional[Dict] = None,
        regime_history: Optional[Dict] = None,
        trade_history: Optional[Dict] = None,
    ) -> DiagnosticReport:
        """Analyze system performance and identify causes of degradation.

        When performance is poor, checks each potential cause and builds
        an evidence-based diagnostic report.

        Args:
            equity_curve: Dict with 'daily_pnl' key containing recent P&L.
            health_history: Dict of health check scores from the health service.
            regime_history: Dict with 'current_regime' and 'regime_duration'.
            trade_history: Dict with recent trade statistics.

        Returns:
            DiagnosticReport with findings and recommendations.
        """
        health_history = health_history or {}
        regime_history = regime_history or {}
        trade_history = trade_history or {}

        diagnostics: List[DiagnosticFinding] = []
        recent_return = None
        recent_sharpe = None

        # ── Check recent performance ──
        is_underperforming = False
        if equity_curve and "daily_pnl" in equity_curve:
            daily_pnl = np.asarray(equity_curve["daily_pnl"], dtype=float)
            recent = daily_pnl[-self.underperformance_window:]
            if len(recent) > 5:
                recent_return = float(np.mean(recent))
                std = float(np.std(recent, ddof=1)) if len(recent) > 1 else 1.0
                recent_sharpe = float(recent_return / std * np.sqrt(252)) if std > 1e-10 else 0.0
                is_underperforming = recent_return < 0

        if not is_underperforming:
            return DiagnosticReport(
                status="PERFORMING",
                diagnostics=[],
                primary_cause="NONE",
                timestamp=datetime.now().isoformat(),
                recent_return=recent_return,
                recent_sharpe=recent_sharpe,
            )

        # ── Check each potential cause ──

        # 1. Alpha decay
        signal_score = health_history.get("signal_decay_score")
        if signal_score is not None and signal_score < self.alpha_decay_threshold:
            diagnostics.append(DiagnosticFinding(
                cause="ALPHA_DECAY",
                severity="HIGH",
                evidence=f"Signal quality health check score {signal_score:.0f} "
                         f"below threshold {self.alpha_decay_threshold:.0f}",
                recommendation="Retrain model with recent data. Consider adding "
                             "new features or removing degraded ones.",
                metric_value=signal_score,
                metric_threshold=self.alpha_decay_threshold,
            ))

        # 2. Unfavorable regime
        current_regime = regime_history.get("current_regime")
        regime_duration = regime_history.get("regime_duration", 0)
        if current_regime == 3:  # high_volatility
            diagnostics.append(DiagnosticFinding(
                cause="UNFAVORABLE_REGIME",
                severity="MEDIUM",
                evidence=f"Currently in high-volatility regime for "
                         f"{regime_duration} days",
                recommendation="System is designed to reduce exposure in high-vol. "
                             "Wait for regime change or reduce position limits.",
                metric_value=float(regime_duration),
            ))
        elif current_regime == 1 and regime_duration > 20:  # extended bear
            diagnostics.append(DiagnosticFinding(
                cause="UNFAVORABLE_REGIME",
                severity="MEDIUM",
                evidence=f"Extended trending_bear regime for {regime_duration} days",
                recommendation="Consider tightening stops and reducing position "
                             "sizes during extended bear trends.",
                metric_value=float(regime_duration),
            ))

        # 3. Execution degradation
        exec_score = health_history.get("execution_quality_score")
        if exec_score is not None and exec_score < self.execution_quality_threshold:
            diagnostics.append(DiagnosticFinding(
                cause="EXECUTION_DEGRADATION",
                severity="MEDIUM",
                evidence=f"Execution quality score {exec_score:.0f} "
                         f"below threshold {self.execution_quality_threshold:.0f}",
                recommendation="Check market impact model calibration. Consider "
                             "reducing trade sizes or using more patient execution.",
                metric_value=exec_score,
                metric_threshold=self.execution_quality_threshold,
            ))

        # 4. Data staleness
        data_freshness = health_history.get("data_freshness_days", 0)
        if data_freshness > self.data_staleness_days:
            diagnostics.append(DiagnosticFinding(
                cause="STALE_DATA",
                severity="HIGH",
                evidence=f"Data is {data_freshness} days old "
                         f"(threshold: {self.data_staleness_days} days)",
                recommendation="Run data refresh pipeline: "
                             "python run_wrds_daily_refresh.py",
                metric_value=float(data_freshness),
                metric_threshold=float(self.data_staleness_days),
            ))

        # 5. Model staleness
        model_age = health_history.get("model_age_days", 0)
        if model_age > self.model_staleness_days:
            diagnostics.append(DiagnosticFinding(
                cause="MODEL_STALE",
                severity="HIGH",
                evidence=f"Model is {model_age} days old "
                         f"(threshold: {self.model_staleness_days} days)",
                recommendation="Retrain model: python run_retrain.py",
                metric_value=float(model_age),
                metric_threshold=float(self.model_staleness_days),
            ))

        # 6. Feature distribution shift
        psi_score = health_history.get("feature_psi_avg", 0)
        if psi_score > 0.25:
            diagnostics.append(DiagnosticFinding(
                cause="FEATURE_SHIFT",
                severity="HIGH",
                evidence=f"Average feature PSI {psi_score:.3f} exceeds 0.25 "
                         f"(significant distribution shift)",
                recommendation="Features have drifted from training distribution. "
                             "Retrain with recent data to recalibrate.",
                metric_value=psi_score,
                metric_threshold=0.25,
            ))

        # 7. Drawdown lockout
        current_dd = health_history.get("current_drawdown", 0.0)
        if current_dd < self.drawdown_threshold:
            diagnostics.append(DiagnosticFinding(
                cause="DRAWDOWN_LOCKOUT",
                severity="MEDIUM",
                evidence=f"Current drawdown {current_dd:.1%} exceeds "
                         f"threshold {self.drawdown_threshold:.1%}. "
                         f"Position sizing is reduced.",
                recommendation="Drawdown governor is actively reducing position "
                             "sizes. Performance will recover as drawdown heals.",
                metric_value=current_dd,
                metric_threshold=self.drawdown_threshold,
            ))

        # 8. Concentration risk
        top_weight = trade_history.get("max_position_weight", 0.0)
        if top_weight > self.concentration_threshold:
            diagnostics.append(DiagnosticFinding(
                cause="CONCENTRATION_RISK",
                severity="LOW",
                evidence=f"Top position weight {top_weight:.1%} exceeds "
                         f"concentration threshold {self.concentration_threshold:.1%}",
                recommendation="Diversify: reduce max_positions limit or add "
                             "correlation-based position sizing.",
                metric_value=top_weight,
                metric_threshold=self.concentration_threshold,
            ))

        # Determine primary cause (highest severity, first in list)
        severity_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        if diagnostics:
            diagnostics.sort(key=lambda d: severity_rank.get(d.severity, 0), reverse=True)
            primary_cause = diagnostics[0].cause
        else:
            primary_cause = "UNKNOWN"

        return DiagnosticReport(
            status="UNDERPERFORMING",
            diagnostics=diagnostics,
            primary_cause=primary_cause,
            timestamp=datetime.now().isoformat(),
            recent_return=recent_return,
            recent_sharpe=recent_sharpe,
        )
