"""
Evaluation Engine — orchestrates all evaluation components into a
comprehensive analysis with red-flag detection and report generation.

Usage::

    engine = EvaluationEngine()
    results = engine.evaluate(
        returns=returns_series,
        predictions=predictions_array,
        trades=trade_list,
        regime_states=regime_array,
    )
    engine.generate_report(results, output_path="results/eval_report.html")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    EVAL_WF_TRAIN_WINDOW,
    EVAL_WF_EMBARGO_DAYS,
    EVAL_WF_TEST_WINDOW,
    EVAL_WF_SLIDE_FREQ,
    EVAL_IC_ROLLING_WINDOW,
    EVAL_IC_DECAY_THRESHOLD,
    EVAL_IC_DECAY_LOOKBACK,
    EVAL_CALIBRATION_BINS,
    EVAL_TOP_N_TRADES,
    EVAL_RECOVERY_WINDOW,
    EVAL_CRITICAL_SLOWING_WINDOW,
    EVAL_REGIME_SHARPE_DIVERGENCE,
    EVAL_OVERFIT_GAP_THRESHOLD,
    EVAL_PNL_CONCENTRATION_THRESHOLD,
    EVAL_CALIBRATION_ERROR_THRESHOLD,
)

from .slicing import PerformanceSlice, SliceRegistry
from .metrics import compute_slice_metrics, decile_spread
from .fragility import (
    pnl_concentration,
    drawdown_distribution,
    recovery_time_distribution,
    detect_critical_slowing_down,
    consecutive_loss_frequency,
)
from .ml_diagnostics import feature_importance_drift, ensemble_disagreement
from .calibration_analysis import analyze_calibration

logger = logging.getLogger(__name__)


@dataclass
class RedFlag:
    """A single red flag detected during evaluation."""
    category: str       # "regime", "overfit", "fragility", "calibration", "ic_decay", "slowing"
    severity: str       # "warning" or "critical"
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation output."""
    # Aggregate metrics
    aggregate_metrics: Dict
    # Per-regime slice metrics
    regime_slice_metrics: Dict[str, Dict]
    individual_regime_metrics: Dict[str, Dict]
    # Walk-forward with embargo
    walk_forward: Optional[Dict] = None
    # Rolling IC
    rolling_ic_data: Optional[Dict] = None
    ic_decay: Optional[Dict] = None
    # Decile spread
    decile_spread_result: Optional[Dict] = None
    # Calibration
    calibration: Optional[Dict] = None
    # Fragility
    fragility: Optional[Dict] = None
    drawdown_dist: Optional[Dict] = None
    recovery_times: Optional[Dict] = None
    critical_slowing: Optional[Dict] = None
    loss_streaks: Optional[Dict] = None
    # ML diagnostics
    feature_drift: Optional[Dict] = None
    ensemble_disagreement_result: Optional[Dict] = None
    # Red flags
    red_flags: List[RedFlag] = field(default_factory=list)
    # Summary
    overall_pass: bool = True
    summary: str = ""


class EvaluationEngine:
    """Orchestrates all evaluation components.

    Parameters
    ----------
    train_window : int
        Walk-forward training window.
    embargo_days : int
        Walk-forward embargo gap.
    test_window : int
        Walk-forward test window.
    slide_freq : str
        ``"weekly"`` or ``"daily"``.
    ic_window : int
        Rolling IC window.
    ic_decay_threshold : float
        IC threshold for decay detection.
    calibration_bins : int
        Number of calibration curve bins.
    top_n_trades : list[int]
        Top-N values for PnL concentration.
    """

    def __init__(
        self,
        train_window: int = EVAL_WF_TRAIN_WINDOW,
        embargo_days: int = EVAL_WF_EMBARGO_DAYS,
        test_window: int = EVAL_WF_TEST_WINDOW,
        slide_freq: str = EVAL_WF_SLIDE_FREQ,
        ic_window: int = EVAL_IC_ROLLING_WINDOW,
        ic_decay_threshold: float = EVAL_IC_DECAY_THRESHOLD,
        calibration_bins: int = EVAL_CALIBRATION_BINS,
        top_n_trades: Optional[List[int]] = None,
    ):
        self.train_window = train_window
        self.embargo_days = embargo_days
        self.test_window = test_window
        self.slide_freq = slide_freq
        self.ic_window = ic_window
        self.ic_decay_threshold = ic_decay_threshold
        self.calibration_bins = calibration_bins
        self.top_n_trades = top_n_trades or list(EVAL_TOP_N_TRADES)

    def evaluate(
        self,
        returns: pd.Series,
        predictions: np.ndarray,
        trades: Optional[List[Dict]] = None,
        regime_states: Optional[np.ndarray] = None,
        confidence_scores: Optional[np.ndarray] = None,
        volatility: Optional[pd.Series] = None,
        importance_matrices: Optional[Dict[str, np.ndarray]] = None,
        ensemble_predictions: Optional[Dict[str, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Run all evaluation analyses.

        Parameters
        ----------
        returns : pd.Series
            Actual return series.
        predictions : np.ndarray
            Model predictions aligned to returns.
        trades : list[dict], optional
            Trade records (for PnL concentration).
        regime_states : np.ndarray, optional
            Regime labels per bar (0-3).
        confidence_scores : np.ndarray, optional
            Model confidence per prediction.
        volatility : pd.Series, optional
            Per-bar volatility for metadata.
        importance_matrices : dict, optional
            Feature importance per retraining period.
        ensemble_predictions : dict, optional
            Per-model predictions for disagreement analysis.
        feature_names : list[str], optional
            Feature names for importance tracking.

        Returns
        -------
        EvaluationResult
        """
        red_flags: List[RedFlag] = []
        pred_arr = np.asarray(predictions, dtype=float).ravel()

        # ── 1. Aggregate metrics ──
        aggregate = compute_slice_metrics(returns, pred_arr)

        # ── 2. Regime slicing ──
        regime_slice_metrics = {}
        individual_regime_metrics = {}

        if regime_states is not None:
            metadata = SliceRegistry.build_metadata(returns, regime_states, volatility)
            primary_slices = SliceRegistry.create_regime_slices(regime_states)
            individual_slices = SliceRegistry.create_individual_regime_slices()

            for slc in primary_slices:
                filtered, info = slc.apply(returns, metadata)
                if info["n_samples"] > 0:
                    metrics = compute_slice_metrics(filtered, None)
                    metrics.update(info)
                    regime_slice_metrics[slc.name] = metrics

            for slc in individual_slices:
                filtered, info = slc.apply(returns, metadata)
                if info["n_samples"] > 0:
                    metrics = compute_slice_metrics(filtered, None)
                    metrics.update(info)
                    individual_regime_metrics[slc.name] = metrics

            # Red flag: regime Sharpes diverge > threshold
            sharpes = [m["sharpe"] for m in individual_regime_metrics.values() if m["n_samples"] >= 20]
            if len(sharpes) >= 2:
                sharpe_range = max(sharpes) - min(sharpes)
                max_abs = max(abs(s) for s in sharpes) if sharpes else 1.0
                if max_abs > 0 and sharpe_range / max_abs > EVAL_REGIME_SHARPE_DIVERGENCE:
                    red_flags.append(RedFlag(
                        category="regime",
                        severity="warning",
                        message=f"Regime Sharpes diverge significantly (range={sharpe_range:.2f})",
                        details={k: v["sharpe"] for k, v in individual_regime_metrics.items()},
                    ))

        # ── 3. Walk-forward with embargo ──
        wf_result = None
        try:
            from ..backtest.validation import walk_forward_with_embargo
            wf = walk_forward_with_embargo(
                returns=returns,
                predictions=pred_arr,
                train_window=self.train_window,
                embargo=self.embargo_days,
                test_window=self.test_window,
                slide_freq=self.slide_freq,
            )
            wf_result = {
                "n_folds": wf.n_folds,
                "mean_train_sharpe": wf.mean_train_sharpe,
                "mean_test_sharpe": wf.mean_test_sharpe,
                "mean_overfit_gap": wf.mean_overfit_gap,
                "is_overfit": wf.is_overfit,
                "warnings": wf.warnings,
                "folds": [
                    {
                        "fold": f.fold,
                        "train_sharpe": f.train_sharpe,
                        "test_sharpe": f.test_sharpe,
                        "train_ic": f.train_ic,
                        "test_ic": f.test_ic,
                        "overfit_gap_sharpe": f.overfit_gap_sharpe,
                        "test_n_samples": f.test_n_samples,
                    }
                    for f in wf.folds
                ],
            }
            if wf.is_overfit:
                red_flags.append(RedFlag(
                    category="overfit",
                    severity="critical",
                    message=f"Walk-forward detects overfitting (gap={wf.mean_overfit_gap:.3f})",
                    details={"mean_gap": wf.mean_overfit_gap, "test_sharpe": wf.mean_test_sharpe},
                ))
        except Exception as e:
            logger.warning("Walk-forward evaluation failed: %s", e)

        # ── 4. Rolling IC and decay ──
        ic_data = None
        ic_decay_result = None
        try:
            from ..backtest.validation import rolling_ic, detect_ic_decay
            ic_series = rolling_ic(pred_arr, returns, window=self.ic_window)
            ic_clean = ic_series.dropna()
            ic_data = {
                "mean_ic": float(ic_clean.mean()) if len(ic_clean) > 0 else 0.0,
                "std_ic": float(ic_clean.std()) if len(ic_clean) > 0 else 0.0,
                "current_ic": float(ic_clean.iloc[-1]) if len(ic_clean) > 0 else 0.0,
                "n_windows": len(ic_clean),
                "ic_series_index": [str(d) for d in ic_clean.index],
                "ic_series_values": ic_clean.values.tolist(),
            }

            decaying, decay_info = detect_ic_decay(
                ic_series, decay_threshold=self.ic_decay_threshold,
            )
            ic_decay_result = {"decaying": decaying, **decay_info}

            if decaying:
                red_flags.append(RedFlag(
                    category="ic_decay",
                    severity="warning",
                    message=f"IC decay detected (mean={decay_info.get('mean_ic', 0):.4f}, slope={decay_info.get('slope', 0):.6f})",
                    details=decay_info,
                ))
        except Exception as e:
            logger.warning("Rolling IC computation failed: %s", e)

        # ── 5. Decile spread ──
        ds_result = None
        try:
            ds_result = decile_spread(
                pred_arr, returns, regime_states=regime_states,
            )
        except Exception as e:
            logger.warning("Decile spread computation failed: %s", e)

        # ── 6. Calibration ──
        cal_result = None
        try:
            cal_result = analyze_calibration(
                pred_arr, returns,
                confidence_scores=confidence_scores,
                bins=self.calibration_bins,
            )
            if cal_result.get("calibration_error", 0) > EVAL_CALIBRATION_ERROR_THRESHOLD:
                red_flags.append(RedFlag(
                    category="calibration",
                    severity="warning",
                    message=f"Poor calibration (error={cal_result['calibration_error']:.4f})",
                    details=cal_result,
                ))
        except Exception as e:
            logger.warning("Calibration analysis failed: %s", e)

        # ── 7. Fragility metrics ──
        frag_result = None
        dd_result = None
        rt_data = None
        cs_result = None
        ls_result = None

        if trades:
            try:
                frag_result = pnl_concentration(trades, self.top_n_trades)
                if frag_result.get("fragile", False):
                    max_n = max(self.top_n_trades)
                    pct = frag_result.get(f"top_{max_n}_pct", 0)
                    red_flags.append(RedFlag(
                        category="fragility",
                        severity="warning",
                        message=f"PnL concentrated: top {max_n} trades = {pct:.0%} of total",
                        details=frag_result,
                    ))
            except Exception as e:
                logger.warning("PnL concentration failed: %s", e)

        try:
            dd_result = drawdown_distribution(returns)
        except Exception as e:
            logger.warning("Drawdown distribution failed: %s", e)

        try:
            rt_series = recovery_time_distribution(returns)
            if len(rt_series) > 0:
                rt_data = {
                    "n_episodes": len(rt_series),
                    "mean": float(rt_series.mean()),
                    "median": float(rt_series.median()),
                    "max": float(rt_series.max()),
                    "values": rt_series.values.tolist(),
                }

                critical, cs_info = detect_critical_slowing_down(rt_series)
                cs_result = {"critical_slowing": critical, **cs_info}
                if critical:
                    red_flags.append(RedFlag(
                        category="slowing",
                        severity="critical",
                        message="Critical slowing down detected — recovery times trending upward",
                        details=cs_info,
                    ))
        except Exception as e:
            logger.warning("Recovery time / critical slowing failed: %s", e)

        try:
            ls_result = consecutive_loss_frequency(returns)
        except Exception as e:
            logger.warning("Loss streak analysis failed: %s", e)

        # ── 8. ML diagnostics ──
        feat_drift = None
        ens_disagree = None

        if importance_matrices:
            try:
                feat_drift = feature_importance_drift(
                    importance_matrices, feature_names=feature_names,
                )
                if feat_drift.get("drift_detected", False):
                    red_flags.append(RedFlag(
                        category="feature_drift",
                        severity="warning",
                        message=f"Feature importance drift detected (min_corr={feat_drift['min_correlation']:.3f})",
                        details=feat_drift,
                    ))
            except Exception as e:
                logger.warning("Feature importance drift failed: %s", e)

        if ensemble_predictions:
            try:
                ens_disagree = ensemble_disagreement(ensemble_predictions)
                if ens_disagree.get("high_disagreement", False):
                    red_flags.append(RedFlag(
                        category="ensemble_disagreement",
                        severity="warning",
                        message=f"Ensemble disagreement detected (min_corr={ens_disagree['min_correlation']:.3f})",
                        details=ens_disagree,
                    ))
            except Exception as e:
                logger.warning("Ensemble disagreement failed: %s", e)

        # ── Overall assessment ──
        critical_flags = [f for f in red_flags if f.severity == "critical"]
        overall_pass = len(critical_flags) == 0

        summary_parts = []
        summary_parts.append(f"Sharpe={aggregate.get('sharpe', 0):.2f}")
        if wf_result:
            summary_parts.append(f"OOS_Sharpe={wf_result['mean_test_sharpe']:.2f}")
        if ic_data:
            summary_parts.append(f"IC={ic_data['mean_ic']:.4f}")
        summary_parts.append(f"RedFlags={len(red_flags)}")
        summary_parts.append("PASS" if overall_pass else "FAIL")
        summary = " | ".join(summary_parts)

        return EvaluationResult(
            aggregate_metrics=aggregate,
            regime_slice_metrics=regime_slice_metrics,
            individual_regime_metrics=individual_regime_metrics,
            walk_forward=wf_result,
            rolling_ic_data=ic_data,
            ic_decay=ic_decay_result,
            decile_spread_result=ds_result,
            calibration=cal_result,
            fragility=frag_result,
            drawdown_dist=dd_result,
            recovery_times=rt_data,
            critical_slowing=cs_result,
            loss_streaks=ls_result,
            feature_drift=feat_drift,
            ensemble_disagreement_result=ens_disagree,
            red_flags=red_flags,
            overall_pass=overall_pass,
            summary=summary,
        )

    def generate_report(
        self,
        results: EvaluationResult,
        output_path: str,
        fmt: str = "html",
    ) -> str:
        """Generate a report from evaluation results.

        Parameters
        ----------
        results : EvaluationResult
            Output from :meth:`evaluate`.
        output_path : str
            File path for the output report.
        fmt : str
            ``"html"`` or ``"json"``.

        Returns
        -------
        str
            Absolute path to the generated report.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            return self._generate_json(results, path)
        return self._generate_html(results, path)

    def _generate_json(self, results: EvaluationResult, path: Path) -> str:
        """Serialize evaluation results to JSON."""
        data = {
            "summary": results.summary,
            "overall_pass": results.overall_pass,
            "aggregate_metrics": results.aggregate_metrics,
            "regime_slice_metrics": results.regime_slice_metrics,
            "individual_regime_metrics": results.individual_regime_metrics,
            "walk_forward": results.walk_forward,
            "rolling_ic": results.rolling_ic_data,
            "ic_decay": results.ic_decay,
            "decile_spread": results.decile_spread_result,
            "calibration": _strip_non_serializable(results.calibration),
            "fragility": results.fragility,
            "drawdown_distribution": results.drawdown_dist,
            "recovery_times": results.recovery_times,
            "critical_slowing": results.critical_slowing,
            "loss_streaks": results.loss_streaks,
            "feature_drift": _strip_non_serializable(results.feature_drift),
            "ensemble_disagreement": results.ensemble_disagreement_result,
            "red_flags": [
                {"category": f.category, "severity": f.severity, "message": f.message}
                for f in results.red_flags
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("JSON report written to %s", path)
        return str(path.resolve())

    def _generate_html(self, results: EvaluationResult, path: Path) -> str:
        """Generate an HTML report with embedded charts."""
        from . import visualization as viz

        sections = []

        # Header
        pass_class = "pass" if results.overall_pass else "fail"
        sections.append(f"""
        <div class="header">
            <h1>Evaluation Report</h1>
            <div class="status {pass_class}">
                {results.summary}
            </div>
        </div>
        """)

        # Red flags
        if results.red_flags:
            flags_html = "<h2>Red Flags</h2><ul>"
            for f in results.red_flags:
                icon = "&#9888;" if f.severity == "critical" else "&#9432;"
                flags_html += f'<li class="{f.severity}">{icon} [{f.category}] {f.message}</li>'
            flags_html += "</ul>"
            sections.append(flags_html)

        # Aggregate metrics
        agg = results.aggregate_metrics
        sections.append(f"""
        <h2>Aggregate Metrics</h2>
        <table>
            <tr><td>Sharpe Ratio</td><td>{agg.get('sharpe', 0):.3f} (&plusmn;{agg.get('sharpe_se', 0):.3f})</td></tr>
            <tr><td>Mean Return</td><td>{agg.get('mean_return', 0):.6f}</td></tr>
            <tr><td>Annualized Return</td><td>{agg.get('annualized_return', 0):.2%}</td></tr>
            <tr><td>Max Drawdown</td><td>{agg.get('max_dd', 0):.2%}</td></tr>
            <tr><td>Win Rate</td><td>{agg.get('win_rate', 0):.1%}</td></tr>
            <tr><td>IC</td><td>{agg.get('ic', 0):.4f}</td></tr>
            <tr><td>N Samples</td><td>{agg.get('n_samples', 0)}</td></tr>
        </table>
        """)

        # Regime slicing
        if results.regime_slice_metrics:
            chart = viz.plot_regime_slices(results.regime_slice_metrics)
            sections.append("<h2>Regime Slice Analysis</h2>")
            if chart.get("html"):
                sections.append(chart["html"])
            sections.append(self._metrics_table(results.regime_slice_metrics))

        if results.individual_regime_metrics:
            chart = viz.plot_regime_slices(results.individual_regime_metrics)
            sections.append("<h2>Per-Regime Metrics</h2>")
            if chart.get("html"):
                sections.append(chart["html"])
            sections.append(self._metrics_table(results.individual_regime_metrics))

        # Walk-forward
        if results.walk_forward:
            wf = results.walk_forward
            sections.append(f"""
            <h2>Walk-Forward Analysis (Embargo={self.embargo_days}d)</h2>
            <table>
                <tr><td>Folds</td><td>{wf['n_folds']}</td></tr>
                <tr><td>Mean IS Sharpe</td><td>{wf['mean_train_sharpe']:.3f}</td></tr>
                <tr><td>Mean OOS Sharpe</td><td>{wf['mean_test_sharpe']:.3f}</td></tr>
                <tr><td>Overfit Gap</td><td>{wf['mean_overfit_gap']:.3f}</td></tr>
                <tr><td>Overfitting?</td><td>{'YES' if wf['is_overfit'] else 'No'}</td></tr>
            </table>
            """)
            if wf.get("folds"):
                chart = viz.plot_walk_forward_folds(wf["folds"])
                if chart.get("html"):
                    sections.append(chart["html"])

        # Rolling IC
        if results.rolling_ic_data:
            ic = results.rolling_ic_data
            sections.append(f"""
            <h2>Rolling Information Coefficient</h2>
            <table>
                <tr><td>Mean IC</td><td>{ic['mean_ic']:.4f}</td></tr>
                <tr><td>Current IC</td><td>{ic['current_ic']:.4f}</td></tr>
                <tr><td>IC Std</td><td>{ic['std_ic']:.4f}</td></tr>
            </table>
            """)
            if ic.get("ic_series_values"):
                ic_series = pd.Series(
                    ic["ic_series_values"],
                    index=pd.to_datetime(ic["ic_series_index"], errors="coerce"),
                )
                chart = viz.plot_rolling_ic(ic_series, self.ic_decay_threshold)
                if chart.get("html"):
                    sections.append(chart["html"])

        # Decile spread
        if results.decile_spread_result:
            ds = results.decile_spread_result
            sections.append(f"""
            <h2>Decile Spread Analysis</h2>
            <table>
                <tr><td>Spread (D10 - D1)</td><td>{ds.get('spread', 0):.6f}</td></tr>
                <tr><td>T-Stat</td><td>{ds.get('spread_t_stat', 0):.3f}</td></tr>
                <tr><td>P-Value</td><td>{ds.get('spread_pvalue', 1):.4f}</td></tr>
                <tr><td>Monotonicity</td><td>{ds.get('monotonicity', 0):.3f}</td></tr>
                <tr><td>Significant</td><td>{'YES' if ds.get('significant') else 'No'}</td></tr>
            </table>
            """)
            chart = viz.plot_decile_spread(
                ds.get("decile_returns", []),
                ds.get("decile_counts"),
            )
            if chart.get("html"):
                sections.append(chart["html"])

        # Calibration
        if results.calibration:
            cal = results.calibration
            sections.append(f"""
            <h2>Calibration Analysis</h2>
            <table>
                <tr><td>ECE</td><td>{cal.get('ece', 0):.4f}</td></tr>
                <tr><td>Calibration Error (MSE)</td><td>{cal.get('calibration_error', 0):.4f}</td></tr>
                <tr><td>Max Gap</td><td>{cal.get('max_gap', 0):.4f}</td></tr>
                <tr><td>Overconfident?</td><td>{'YES' if cal.get('overconfident') else 'No'}</td></tr>
            </table>
            """)
            if cal.get("reliability_curve"):
                chart = viz.plot_calibration_curve(cal["reliability_curve"])
                if chart.get("html"):
                    sections.append(chart["html"])

        # Fragility
        if results.fragility or results.drawdown_dist:
            sections.append("<h2>Fragility Analysis</h2>")

            if results.fragility:
                frag = results.fragility
                rows = ""
                for n_val in self.top_n_trades:
                    key = f"top_{n_val}_pct"
                    rows += f"<tr><td>Top {n_val} PnL %</td><td>{frag.get(key, 0):.1%}</td></tr>"
                rows += f"<tr><td>HHI</td><td>{frag.get('herfindahl_index', 0):.4f}</td></tr>"
                rows += f"<tr><td>Fragile?</td><td>{'YES' if frag.get('fragile') else 'No'}</td></tr>"
                sections.append(f"<table>{rows}</table>")

            if results.drawdown_dist:
                dd = results.drawdown_dist
                sections.append(f"""
                <table>
                    <tr><td>Max DD</td><td>{dd.get('max_dd', 0):.2%}</td></tr>
                    <tr><td>Worst Single Day</td><td>{dd.get('max_dd_single_day', 0):.2%}</td></tr>
                    <tr><td># Episodes</td><td>{dd.get('n_episodes', 0)}</td></tr>
                    <tr><td>DD Concentration</td><td>{dd.get('dd_concentration', 0):.1%}</td></tr>
                    <tr><td>% Time Underwater</td><td>{dd.get('pct_time_underwater', 0):.1%}</td></tr>
                    <tr><td>Max DD Duration</td><td>{dd.get('max_dd_duration', 0)} bars</td></tr>
                </table>
                """)

        # Critical slowing
        if results.critical_slowing:
            cs = results.critical_slowing
            sections.append(f"""
            <h2>Critical Slowing Down</h2>
            <table>
                <tr><td>Detected?</td><td>{'YES' if cs.get('critical_slowing') else 'No'}</td></tr>
                <tr><td>Slope</td><td>{cs.get('slope', 0):.4f}</td></tr>
                <tr><td>Recent Trend</td><td>{cs.get('recent_trend', 'N/A')}</td></tr>
                <tr><td>Current Recovery Time</td><td>{cs.get('current_recovery_time', 0):.1f} bars</td></tr>
                <tr><td>Historical Median</td><td>{cs.get('historical_median', 0):.1f} bars</td></tr>
            </table>
            """)

        # Write HTML
        html = _HTML_TEMPLATE.format(content="\n".join(sections))
        path.write_text(html, encoding="utf-8")

        logger.info("HTML report written to %s", path)
        return str(path.resolve())

    @staticmethod
    def _metrics_table(metrics_dict: Dict[str, Dict]) -> str:
        """Build an HTML table from a dict of slice metrics."""
        if not metrics_dict:
            return ""

        header = "<tr><th>Slice</th><th>Sharpe</th><th>Mean Ret</th><th>Max DD</th><th>Win Rate</th><th>N</th><th>Confidence</th></tr>"
        rows = []
        for name, m in metrics_dict.items():
            conf = m.get("confidence", "low")
            conf_class = "low-conf" if conf == "low" else ""
            rows.append(
                f'<tr class="{conf_class}">'
                f"<td>{name}</td>"
                f"<td>{m.get('sharpe', 0):.3f}</td>"
                f"<td>{m.get('mean_return', 0):.6f}</td>"
                f"<td>{m.get('max_dd', 0):.2%}</td>"
                f"<td>{m.get('win_rate', 0):.1%}</td>"
                f"<td>{m.get('n_samples', 0)}</td>"
                f"<td>{conf}</td>"
                f"</tr>"
            )
        return f"<table>{header}{''.join(rows)}</table>"


# ── HTML template ────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evaluation Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
    .header {{ margin-bottom: 30px; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 30px; }}
    .status {{ padding: 10px 15px; border-radius: 5px; font-weight: bold; font-family: monospace; }}
    .status.pass {{ background: #d4edda; color: #155724; }}
    .status.fail {{ background: #f8d7da; color: #721c24; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #3498db; color: white; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    tr.low-conf {{ opacity: 0.6; font-style: italic; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ padding: 8px 12px; margin: 5px 0; border-radius: 4px; }}
    li.critical {{ background: #f8d7da; color: #721c24; }}
    li.warning {{ background: #fff3cd; color: #856404; }}
</style>
</head>
<body>
{content}
</body>
</html>"""


def _strip_non_serializable(d: Any) -> Any:
    """Recursively strip non-JSON-serializable values from a dict."""
    if d is None:
        return None
    if isinstance(d, dict):
        return {k: _strip_non_serializable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_strip_non_serializable(v) for v in d]
    if isinstance(d, (np.integer, np.int64)):
        return int(d)
    if isinstance(d, (np.floating, np.float64)):
        return float(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d
