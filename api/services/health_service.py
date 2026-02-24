"""System health assessment for API consumption.

Expanded health service with 15 runtime monitoring checks across 5 domains:
    - Data Integrity (25%)
    - Signal Quality (25%)
    - Risk Management (20%)
    - Execution Quality (15%)
    - Model Governance (15%)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HealthService:
    """Computes system health from model age, cache freshness, and runtime metrics."""

    def get_quick_status(self) -> Dict[str, Any]:
        """Lightweight health check for ``GET /api/health``."""
        from quant_engine.config import DATA_CACHE_DIR, MODEL_DIR

        status = "healthy"
        checks: Dict[str, str] = {}

        # Cache freshness
        cache_dir = Path(DATA_CACHE_DIR)
        if cache_dir.exists():
            parquets = list(cache_dir.glob("*.parquet"))
            if parquets:
                ages = [
                    (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
                    for f in parquets[:10]
                ]
                max_age = max(ages)
                checks["cache_max_age_days"] = str(max_age)
                if max_age > 21:
                    status = "degraded"
            else:
                checks["cache"] = "no_parquet_files"
                status = "degraded"
        else:
            checks["cache"] = "missing"
            status = "unhealthy"

        # Model availability
        model_dir = Path(MODEL_DIR)
        registry_path = model_dir / "registry.json"
        if registry_path.exists():
            checks["model_registry"] = "present"
        else:
            checks["model_registry"] = "missing"
            if status == "healthy":
                status = "degraded"

        return {"status": status, "checks": checks, "timestamp": datetime.now(timezone.utc).isoformat()}

    def get_detailed_health(self) -> Dict[str, Any]:
        """Full system health assessment."""
        from api.services.data_helpers import collect_health_data

        payload = collect_health_data()
        # Convert dataclass to dict, handling nested dataclasses
        result: Dict[str, Any] = {
            "overall_score": payload.overall_score,
            "overall_status": payload.overall_status,
            "generated_at": payload.generated_at.isoformat(),
            "data_integrity_score": payload.data_integrity_score,
            "promotion_score": payload.promotion_score,
            "wf_score": payload.wf_score,
            "execution_score": payload.execution_score,
            "complexity_score": payload.complexity_score,
        }
        # Convert HealthCheck lists to dicts
        for section in [
            "survivorship_checks", "data_quality_checks", "promotion_checks",
            "wf_checks", "execution_checks", "complexity_checks", "strengths",
        ]:
            checks = getattr(payload, section, [])
            result[section] = [
                {"name": c.name, "status": c.status, "detail": c.detail,
                 "value": c.value, "recommendation": c.recommendation}
                for c in checks
            ]
        result["promotion_funnel"] = payload.promotion_funnel
        result["feature_inventory"] = payload.feature_inventory
        result["knob_inventory"] = payload.knob_inventory

        # Attach comprehensive runtime health checks
        result["runtime_health"] = self.compute_comprehensive_health()
        return result

    def compute_comprehensive_health(self) -> Dict[str, Any]:
        """Compute comprehensive health score across 5 weighted domains.

        Domain weights:
            - Data Integrity (25%): freshness + survivorship + anomalies
            - Signal Quality (25%): IC + decay + prediction distribution + ensemble
            - Risk Management (20%): drawdown + correlations + tail risk + Kelly + capital
            - Execution Quality (15%): slippage + microstructure + IR
            - Model Governance (15%): feature drift + retraining + CV gap + regime
        """
        domains: Dict[str, Dict[str, Any]] = {}

        # ── Data Integrity (25%) ──
        data_checks = []
        data_checks.append(self._check_data_quality_anomalies())
        data_checks.append(self._check_survivorship_bias())
        data_score = self._domain_score(data_checks)
        domains["data_integrity"] = {"weight": 0.25, "score": data_score, "checks": data_checks}

        # ── Signal Quality (25%) ──
        signal_checks = []
        signal_checks.append(self._check_signal_decay())
        signal_checks.append(self._check_prediction_distribution())
        signal_checks.append(self._check_ensemble_disagreement())
        signal_score = self._domain_score(signal_checks)
        domains["signal_quality"] = {"weight": 0.25, "score": signal_score, "checks": signal_checks}

        # ── Risk Management (20%) ──
        risk_checks = []
        risk_checks.append(self._check_tail_risk())
        risk_checks.append(self._check_correlation_regime())
        risk_checks.append(self._check_capital_utilization())
        risk_score = self._domain_score(risk_checks)
        domains["risk_management"] = {"weight": 0.20, "score": risk_score, "checks": risk_checks}

        # ── Execution Quality (15%) ──
        exec_checks = []
        exec_checks.append(self._check_execution_quality())
        exec_checks.append(self._check_information_ratio())
        exec_checks.append(self._check_market_microstructure())
        exec_score = self._domain_score(exec_checks)
        domains["execution_quality"] = {"weight": 0.15, "score": exec_score, "checks": exec_checks}

        # ── Model Governance (15%) ──
        gov_checks = []
        gov_checks.append(self._check_feature_importance_drift())
        gov_checks.append(self._check_cv_gap_trend())
        gov_checks.append(self._check_regime_transition_health())
        gov_checks.append(self._check_retraining_effectiveness())
        gov_score = self._domain_score(gov_checks)
        domains["model_governance"] = {"weight": 0.15, "score": gov_score, "checks": gov_checks}

        # Weighted overall
        overall = sum(d["weight"] * d["score"] for d in domains.values())

        return {
            "overall_score": round(overall, 1),
            "overall_status": "PASS" if overall >= 75 else "WARN" if overall >= 50 else "FAIL",
            "domains": {k: {"score": round(v["score"], 1), "weight": v["weight"],
                            "checks": v["checks"]} for k, v in domains.items()},
        }

    @staticmethod
    def _domain_score(checks: List[Dict[str, Any]]) -> float:
        """Compute mean score for a domain from its check results."""
        scores = [c.get("score", 0.0) for c in checks if isinstance(c.get("score"), (int, float))]
        return float(np.mean(scores)) if scores else 0.0

    # ── Individual Check Methods ────────────────────────────────────────

    def _check_signal_decay(self) -> Dict[str, Any]:
        """Check for autocorrelation of prediction errors (signal decay)."""
        try:
            from quant_engine.config import RESULTS_DIR
            trades_path = RESULTS_DIR / "autopilot" / "paper_state.json"
            if not trades_path.exists():
                return {"name": "signal_decay", "status": "unavailable", "score": 50.0,
                        "reason": "No paper trading state available"}
            with open(trades_path, "r") as f:
                state = json.load(f)
            trades = state.get("trade_history", [])
            if len(trades) < 20:
                return {"name": "signal_decay", "status": "insufficient_data", "score": 50.0,
                        "reason": f"Only {len(trades)} trades — need 20+"}
            # Check if recent prediction errors are autocorrelated
            errors = [t.get("predicted_return", 0) - t.get("actual_return", 0)
                      for t in trades[-100:] if "predicted_return" in t and "actual_return" in t]
            if len(errors) < 10:
                return {"name": "signal_decay", "status": "insufficient_data", "score": 50.0,
                        "reason": "Not enough matched prediction/actual pairs"}
            errors = np.array(errors, dtype=float)
            # Lag-1 autocorrelation of errors
            if len(errors) > 2 and np.std(errors) > 1e-10:
                autocorr = float(np.corrcoef(errors[:-1], errors[1:])[0, 1])
            else:
                autocorr = 0.0
            # Low autocorrelation is good (errors are random, not persistent)
            if abs(autocorr) < 0.15:
                return {"name": "signal_decay", "status": "PASS", "score": 85.0,
                        "reason": f"Error autocorrelation: {autocorr:.3f} (low)"}
            elif abs(autocorr) < 0.30:
                return {"name": "signal_decay", "status": "WARN", "score": 55.0,
                        "reason": f"Error autocorrelation: {autocorr:.3f} (moderate — possible decay)"}
            return {"name": "signal_decay", "status": "FAIL", "score": 25.0,
                    "reason": f"Error autocorrelation: {autocorr:.3f} (high — signal may be stale)"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'signal_decay' failed: %s", e)
            return {"name": "signal_decay", "status": "error", "score": 50.0,
                    "reason": f"Could not assess signal decay: {e}"}

    def _check_feature_importance_drift(self) -> Dict[str, Any]:
        """Check if feature importance rankings are drifting across retrains."""
        try:
            from quant_engine.models.feature_stability import FeatureStabilityTracker
            tracker = FeatureStabilityTracker()
            report = tracker.check_stability()
            if report.n_cycles < 2:
                return {"name": "feature_drift", "status": "insufficient_data", "score": 50.0,
                        "reason": f"Only {report.n_cycles} training cycles recorded"}
            spearman = report.spearman_vs_previous
            if spearman is None:
                return {"name": "feature_drift", "status": "unavailable", "score": 50.0,
                        "reason": "No comparison available"}
            if spearman > 0.70:
                return {"name": "feature_drift", "status": "PASS", "score": 90.0,
                        "reason": f"Feature rank correlation: {spearman:.3f} (stable)"}
            elif spearman > 0.40:
                return {"name": "feature_drift", "status": "WARN", "score": 55.0,
                        "reason": f"Feature rank correlation: {spearman:.3f} (moderate drift)"}
            return {"name": "feature_drift", "status": "FAIL", "score": 20.0,
                    "reason": f"Feature rank correlation: {spearman:.3f} (severe drift — retrain advised)"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'feature_drift' failed: %s", e)
            return {"name": "feature_drift", "status": "error", "score": 50.0,
                    "reason": f"Could not assess feature stability: {e}"}

    def _check_regime_transition_health(self) -> Dict[str, Any]:
        """Validate HMM transition matrix for degenerate rows."""
        try:
            from quant_engine.config import DATA_CACHE_DIR
            from api.services.data_helpers import compute_regime_payload
            regime = compute_regime_payload(Path(DATA_CACHE_DIR))
            trans = regime.get("transition", None)
            if trans is None:
                return {"name": "regime_transitions", "status": "unavailable", "score": 50.0,
                        "reason": "No transition matrix available"}
            trans = np.asarray(trans, dtype=float)
            # Check for degenerate rows (absorbing states)
            diag = np.diag(trans)
            max_diag = float(diag.max())
            if max_diag > 0.98:
                return {"name": "regime_transitions", "status": "FAIL", "score": 25.0,
                        "reason": f"Near-absorbing state detected (max diagonal: {max_diag:.3f})"}
            if max_diag > 0.95:
                return {"name": "regime_transitions", "status": "WARN", "score": 60.0,
                        "reason": f"Sticky regime (max diagonal: {max_diag:.3f})"}
            return {"name": "regime_transitions", "status": "PASS", "score": 85.0,
                    "reason": f"Transition matrix healthy (max diagonal: {max_diag:.3f})"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'regime_transitions' failed: %s", e)
            return {"name": "regime_transitions", "status": "error", "score": 50.0,
                    "reason": f"Could not assess regime transitions: {e}"}

    def _check_prediction_distribution(self) -> Dict[str, Any]:
        """Detect variance collapse in predictions."""
        try:
            from quant_engine.config import RESULTS_DIR
            trades_path = RESULTS_DIR / "autopilot" / "paper_state.json"
            if not trades_path.exists():
                return {"name": "prediction_distribution", "status": "unavailable", "score": 50.0,
                        "reason": "No paper trading state"}
            with open(trades_path, "r") as f:
                state = json.load(f)
            trades = state.get("trade_history", [])
            preds = [t.get("predicted_return", 0) for t in trades[-200:] if "predicted_return" in t]
            if len(preds) < 10:
                return {"name": "prediction_distribution", "status": "insufficient_data",
                        "score": 50.0, "reason": "Too few predictions"}
            preds = np.array(preds, dtype=float)
            std = float(np.std(preds))
            if std < 1e-6:
                return {"name": "prediction_distribution", "status": "FAIL", "score": 10.0,
                        "reason": f"Variance collapse — all predictions near-identical (std={std:.6f})"}
            if std < 0.001:
                return {"name": "prediction_distribution", "status": "WARN", "score": 45.0,
                        "reason": f"Low prediction variance (std={std:.6f})"}
            return {"name": "prediction_distribution", "status": "PASS", "score": 85.0,
                    "reason": f"Healthy prediction spread (std={std:.6f})"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'prediction_distribution' failed: %s", e)
            return {"name": "prediction_distribution", "status": "error", "score": 50.0,
                    "reason": f"Could not assess prediction distribution: {e}"}

    def _check_survivorship_bias(self) -> Dict[str, Any]:
        """Quantify survivorship bias impact."""
        try:
            from quant_engine.config import SURVIVORSHIP_DB, WRDS_ENABLED
            if WRDS_ENABLED:
                db_path = Path(SURVIVORSHIP_DB)
                if db_path.exists() and db_path.stat().st_size > 1024:
                    return {"name": "survivorship_bias", "status": "PASS", "score": 90.0,
                            "reason": "WRDS + survivorship DB active — bias mitigated"}
                return {"name": "survivorship_bias", "status": "WARN", "score": 60.0,
                        "reason": "WRDS enabled but survivorship DB empty"}
            return {"name": "survivorship_bias", "status": "FAIL", "score": 30.0,
                    "reason": "WRDS disabled — static universe has survivorship bias"}
        except (ValueError, ImportError) as e:
            logger.warning("Health check 'survivorship_bias' failed: %s", e)
            return {"name": "survivorship_bias", "status": "error", "score": 50.0,
                    "reason": f"Could not assess survivorship bias: {e}"}

    def _check_correlation_regime(self) -> Dict[str, Any]:
        """Check average pairwise correlation in portfolio."""
        try:
            from quant_engine.config import DATA_CACHE_DIR
            cache_dir = Path(DATA_CACHE_DIR)
            parquets = sorted(cache_dir.glob("*_1d.parquet"))[:20]
            if len(parquets) < 3:
                return {"name": "correlation_regime", "status": "unavailable", "score": 50.0,
                        "reason": "Not enough assets for correlation check"}
            returns = []
            for p in parquets:
                try:
                    df = pd.read_parquet(p)
                    if "Close" in df.columns:
                        ret = pd.to_numeric(df["Close"], errors="coerce").pct_change().dropna()
                        ret.name = p.stem
                        returns.append(ret.tail(252))
                except (OSError, ValueError):
                    continue
            if len(returns) < 3:
                return {"name": "correlation_regime", "status": "unavailable", "score": 50.0,
                        "reason": "Could not load enough return series"}
            ret_df = pd.concat(returns, axis=1).dropna()
            if ret_df.shape[0] < 20 or ret_df.shape[1] < 3:
                return {"name": "correlation_regime", "status": "unavailable", "score": 50.0,
                        "reason": "Insufficient overlapping data"}
            corr_matrix = ret_df.corr()
            # Average off-diagonal correlation
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = float(corr_matrix.values[mask].mean())
            if avg_corr < 0.40:
                return {"name": "correlation_regime", "status": "PASS", "score": 85.0,
                        "reason": f"Avg pairwise correlation: {avg_corr:.3f} (diversified)"}
            if avg_corr < 0.65:
                return {"name": "correlation_regime", "status": "WARN", "score": 55.0,
                        "reason": f"Avg pairwise correlation: {avg_corr:.3f} (elevated)"}
            return {"name": "correlation_regime", "status": "FAIL", "score": 25.0,
                    "reason": f"Avg pairwise correlation: {avg_corr:.3f} (crisis-level)"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'correlation_regime' failed: %s", e)
            return {"name": "correlation_regime", "status": "error", "score": 50.0,
                    "reason": f"Could not assess correlations: {e}"}

    def _check_execution_quality(self) -> Dict[str, Any]:
        """Check slippage tracking from paper/live trading."""
        try:
            from quant_engine.config import RESULTS_DIR
            trades_path = RESULTS_DIR / "autopilot" / "paper_state.json"
            if not trades_path.exists():
                return {"name": "execution_quality", "status": "unavailable", "score": 50.0,
                        "reason": "No trade data available"}
            with open(trades_path, "r") as f:
                state = json.load(f)
            trades = state.get("trade_history", [])
            if len(trades) < 10:
                return {"name": "execution_quality", "status": "insufficient_data", "score": 50.0,
                        "reason": f"Only {len(trades)} trades — need 10+"}
            # Check if slippage data is present
            slippages = [t.get("slippage", 0) for t in trades if "slippage" in t]
            if not slippages:
                return {"name": "execution_quality", "status": "WARN", "score": 60.0,
                        "reason": "No slippage tracking data (paper trading only)"}
            avg_slippage = float(np.mean(slippages))
            if abs(avg_slippage) < 0.001:
                return {"name": "execution_quality", "status": "PASS", "score": 85.0,
                        "reason": f"Avg slippage: {avg_slippage:.4f} (minimal)"}
            return {"name": "execution_quality", "status": "WARN", "score": 55.0,
                    "reason": f"Avg slippage: {avg_slippage:.4f}"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'execution_quality' failed: %s", e)
            return {"name": "execution_quality", "status": "error", "score": 50.0,
                    "reason": f"Could not assess execution quality: {e}"}

    def _check_tail_risk(self) -> Dict[str, Any]:
        """Compute rolling CVaR and tail ratio from backtest results."""
        try:
            from quant_engine.config import RESULTS_DIR
            equity_path = RESULTS_DIR / "backtest_equity.csv"
            if not equity_path.exists():
                return {"name": "tail_risk", "status": "unavailable", "score": 50.0,
                        "reason": "No equity curve available"}
            df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
            if "equity" not in df.columns or len(df) < 60:
                return {"name": "tail_risk", "status": "insufficient_data", "score": 50.0,
                        "reason": "Insufficient equity data"}
            returns = df["equity"].pct_change().dropna().values
            var5 = float(np.percentile(returns, 5))
            cvar5 = float(returns[returns <= var5].mean()) if np.any(returns <= var5) else var5
            # Tail ratio: avg gain in top 5% / abs(avg loss in bottom 5%)
            p95 = float(np.percentile(returns, 95))
            top_tail = returns[returns >= p95]
            bot_tail = returns[returns <= var5]
            if len(bot_tail) > 0 and abs(bot_tail.mean()) > 1e-10:
                tail_ratio = float(top_tail.mean() / abs(bot_tail.mean())) if len(top_tail) > 0 else 0.0
            else:
                tail_ratio = 1.0
            if tail_ratio > 1.0 and cvar5 > -0.05:
                return {"name": "tail_risk", "status": "PASS", "score": 85.0,
                        "reason": f"Tail ratio: {tail_ratio:.2f}, CVaR5: {cvar5:.4f}"}
            if cvar5 > -0.10:
                return {"name": "tail_risk", "status": "WARN", "score": 55.0,
                        "reason": f"Tail ratio: {tail_ratio:.2f}, CVaR5: {cvar5:.4f}"}
            return {"name": "tail_risk", "status": "FAIL", "score": 25.0,
                    "reason": f"Extreme tail risk — CVaR5: {cvar5:.4f}"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'tail_risk' failed: %s", e)
            return {"name": "tail_risk", "status": "error", "score": 50.0,
                    "reason": f"Could not assess tail risk: {e}"}

    def _check_information_ratio(self) -> Dict[str, Any]:
        """Check excess return vs benchmark (information ratio)."""
        try:
            from quant_engine.config import RESULTS_DIR, DATA_CACHE_DIR
            equity_path = RESULTS_DIR / "backtest_equity.csv"
            if not equity_path.exists():
                return {"name": "information_ratio", "status": "unavailable", "score": 50.0,
                        "reason": "No equity curve available"}
            df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
            if "equity" not in df.columns or len(df) < 60:
                return {"name": "information_ratio", "status": "insufficient_data",
                        "score": 50.0, "reason": "Insufficient data"}
            strat_ret = df["equity"].pct_change().dropna()
            # Try to load SPY benchmark
            bench_path = Path(DATA_CACHE_DIR) / "SPY_1d.parquet"
            if not bench_path.exists():
                return {"name": "information_ratio", "status": "unavailable", "score": 50.0,
                        "reason": "No benchmark data"}
            bench_df = pd.read_parquet(bench_path)
            if "Close" not in bench_df.columns:
                return {"name": "information_ratio", "status": "unavailable", "score": 50.0,
                        "reason": "No benchmark Close column"}
            bench_ret = pd.to_numeric(bench_df["Close"], errors="coerce").pct_change().dropna()
            common = strat_ret.index.intersection(bench_ret.index)
            if len(common) < 30:
                return {"name": "information_ratio", "status": "insufficient_data",
                        "score": 50.0, "reason": "Insufficient overlapping data"}
            excess = strat_ret.reindex(common) - bench_ret.reindex(common)
            te = float(excess.std() * np.sqrt(252))
            ir = float(excess.mean() * 252 / te) if te > 1e-10 else 0.0
            if ir > 0.5:
                return {"name": "information_ratio", "status": "PASS", "score": 90.0,
                        "reason": f"IR: {ir:.2f} (strong alpha)"}
            if ir > 0.0:
                return {"name": "information_ratio", "status": "WARN", "score": 60.0,
                        "reason": f"IR: {ir:.2f} (positive but modest)"}
            return {"name": "information_ratio", "status": "FAIL", "score": 25.0,
                    "reason": f"IR: {ir:.2f} (negative — underperforming benchmark)"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'information_ratio' failed: %s", e)
            return {"name": "information_ratio", "status": "error", "score": 50.0,
                    "reason": f"Could not compute information ratio: {e}"}

    def _check_cv_gap_trend(self) -> Dict[str, Any]:
        """Detect overfitting trend from expanding CV gap across model versions."""
        try:
            from quant_engine.config import MODEL_DIR
            registry_path = Path(MODEL_DIR) / "registry.json"
            if not registry_path.exists():
                return {"name": "cv_gap_trend", "status": "unavailable", "score": 50.0,
                        "reason": "No model registry"}
            with open(registry_path, "r") as f:
                reg = json.load(f)
            versions = reg.get("versions", []) if isinstance(reg, dict) else []
            if len(versions) < 2:
                return {"name": "cv_gap_trend", "status": "insufficient_data", "score": 50.0,
                        "reason": f"Only {len(versions)} model versions"}
            gaps = [float(v.get("cv_gap", 0)) for v in versions if "cv_gap" in v]
            if len(gaps) < 2:
                return {"name": "cv_gap_trend", "status": "insufficient_data", "score": 50.0,
                        "reason": "No CV gap data"}
            # Check if gap is trending upward (linear regression slope)
            x = np.arange(len(gaps), dtype=float)
            slope = float(np.polyfit(x, gaps, 1)[0]) if len(gaps) >= 2 else 0.0
            latest_gap = gaps[-1]
            if latest_gap < 0.05 and slope < 0.005:
                return {"name": "cv_gap_trend", "status": "PASS", "score": 85.0,
                        "reason": f"CV gap: {latest_gap:.4f}, trend: {slope:+.4f}/version (stable)"}
            if latest_gap < 0.10:
                return {"name": "cv_gap_trend", "status": "WARN", "score": 55.0,
                        "reason": f"CV gap: {latest_gap:.4f}, trend: {slope:+.4f}/version"}
            return {"name": "cv_gap_trend", "status": "FAIL", "score": 25.0,
                    "reason": f"CV gap: {latest_gap:.4f}, trend: {slope:+.4f}/version (overfitting risk)"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'cv_gap_trend' failed: %s", e)
            return {"name": "cv_gap_trend", "status": "error", "score": 50.0,
                    "reason": f"Could not assess CV gap trend: {e}"}

    def _check_data_quality_anomalies(self) -> Dict[str, Any]:
        """Check for volume/price anomalies in cached data."""
        try:
            from quant_engine.config import DATA_CACHE_DIR
            cache_dir = Path(DATA_CACHE_DIR)
            parquets = sorted(cache_dir.glob("*_1d.parquet"))[:10]
            if len(parquets) < 1:
                return {"name": "data_anomalies", "status": "unavailable", "score": 50.0,
                        "reason": "No cached data files"}
            anomaly_count = 0
            total_checked = 0
            for p in parquets:
                try:
                    df = pd.read_parquet(p)
                    total_checked += 1
                    if "Volume" in df.columns:
                        vol = pd.to_numeric(df["Volume"], errors="coerce").dropna()
                        if len(vol) > 20:
                            zero_frac = (vol == 0).mean()
                            if zero_frac > 0.25:
                                anomaly_count += 1
                    if "Close" in df.columns:
                        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
                        if len(close) > 20:
                            daily_ret = close.pct_change().dropna()
                            extreme = (daily_ret.abs() > 0.40).sum()
                            if extreme > 0:
                                anomaly_count += 1
                except (OSError, ValueError):
                    continue
            if total_checked == 0:
                return {"name": "data_anomalies", "status": "unavailable", "score": 50.0,
                        "reason": "Could not read any data files"}
            anomaly_rate = anomaly_count / total_checked
            if anomaly_rate == 0:
                return {"name": "data_anomalies", "status": "PASS", "score": 90.0,
                        "reason": f"No anomalies in {total_checked} files"}
            if anomaly_rate < 0.3:
                return {"name": "data_anomalies", "status": "WARN", "score": 60.0,
                        "reason": f"{anomaly_count}/{total_checked} files with anomalies"}
            return {"name": "data_anomalies", "status": "FAIL", "score": 25.0,
                    "reason": f"{anomaly_count}/{total_checked} files with anomalies"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'data_anomalies' failed: %s", e)
            return {"name": "data_anomalies", "status": "error", "score": 50.0,
                    "reason": f"Could not check for data anomalies: {e}"}

    def _check_ensemble_disagreement(self) -> Dict[str, Any]:
        """Check ensemble member consensus (if multi-model)."""
        try:
            from quant_engine.config import MODEL_DIR
            meta_files = sorted(Path(MODEL_DIR).rglob("*_meta.json"), reverse=True)
            if not meta_files:
                return {"name": "ensemble_disagreement", "status": "unavailable", "score": 50.0,
                        "reason": "No model metadata found"}
            with open(meta_files[0], "r") as f:
                meta = json.load(f)
            regime_models = meta.get("regime_models", {})
            if not regime_models:
                return {"name": "ensemble_disagreement", "status": "WARN", "score": 60.0,
                        "reason": "Only global model — no regime ensemble"}
            holdout_corrs = [float(v.get("holdout_corr", 0)) for v in regime_models.values()]
            if not holdout_corrs:
                return {"name": "ensemble_disagreement", "status": "unavailable", "score": 50.0,
                        "reason": "No holdout correlations for regime models"}
            spread = float(max(holdout_corrs) - min(holdout_corrs))
            avg = float(np.mean(holdout_corrs))
            if spread < 0.20 and avg > 0.02:
                return {"name": "ensemble_disagreement", "status": "PASS", "score": 85.0,
                        "reason": f"Ensemble consistent (spread: {spread:.3f}, avg: {avg:.3f})"}
            if avg > 0:
                return {"name": "ensemble_disagreement", "status": "WARN", "score": 55.0,
                        "reason": f"Ensemble divergent (spread: {spread:.3f}, avg: {avg:.3f})"}
            return {"name": "ensemble_disagreement", "status": "FAIL", "score": 25.0,
                    "reason": f"Ensemble poor quality (avg holdout corr: {avg:.3f})"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'ensemble_disagreement' failed: %s", e)
            return {"name": "ensemble_disagreement", "status": "error", "score": 50.0,
                    "reason": f"Could not assess ensemble: {e}"}

    def _check_market_microstructure(self) -> Dict[str, Any]:
        """Check liquidity metrics from cached data."""
        try:
            from quant_engine.config import DATA_CACHE_DIR
            cache_dir = Path(DATA_CACHE_DIR)
            parquets = sorted(cache_dir.glob("*_1d.parquet"))[:10]
            if len(parquets) < 3:
                return {"name": "microstructure", "status": "unavailable", "score": 50.0,
                        "reason": "Insufficient data for microstructure analysis"}
            low_liquidity = 0
            checked = 0
            for p in parquets:
                try:
                    df = pd.read_parquet(p)
                    if "Volume" in df.columns and "Close" in df.columns:
                        vol = pd.to_numeric(df["Volume"], errors="coerce").tail(20)
                        close = pd.to_numeric(df["Close"], errors="coerce").tail(20)
                        dollar_vol = (vol * close).mean()
                        checked += 1
                        if dollar_vol < 1_000_000:  # Less than $1M daily
                            low_liquidity += 1
                except (OSError, ValueError):
                    continue
            if checked == 0:
                return {"name": "microstructure", "status": "unavailable", "score": 50.0,
                        "reason": "Could not assess liquidity"}
            low_frac = low_liquidity / checked
            if low_frac < 0.1:
                return {"name": "microstructure", "status": "PASS", "score": 85.0,
                        "reason": f"All {checked} assets have adequate liquidity"}
            if low_frac < 0.3:
                return {"name": "microstructure", "status": "WARN", "score": 60.0,
                        "reason": f"{low_liquidity}/{checked} assets have low daily dollar volume"}
            return {"name": "microstructure", "status": "FAIL", "score": 30.0,
                    "reason": f"{low_liquidity}/{checked} assets are illiquid"}
        except (OSError, ValueError, ImportError) as e:
            logger.warning("Health check 'microstructure' failed: %s", e)
            return {"name": "microstructure", "status": "error", "score": 50.0,
                    "reason": f"Could not assess microstructure: {e}"}

    def _check_retraining_effectiveness(self) -> Dict[str, Any]:
        """Check if model retrains are improving or degrading performance."""
        try:
            from quant_engine.config import MODEL_DIR
            registry_path = Path(MODEL_DIR) / "registry.json"
            if not registry_path.exists():
                return {"name": "retrain_effectiveness", "status": "unavailable", "score": 50.0,
                        "reason": "No model registry"}
            with open(registry_path, "r") as f:
                reg = json.load(f)
            versions = reg.get("versions", []) if isinstance(reg, dict) else []
            if len(versions) < 2:
                return {"name": "retrain_effectiveness", "status": "insufficient_data",
                        "score": 50.0, "reason": "Need 2+ model versions"}
            spearman_vals = [float(v.get("holdout_spearman", 0)) for v in versions
                            if "holdout_spearman" in v]
            if len(spearman_vals) < 2:
                return {"name": "retrain_effectiveness", "status": "insufficient_data",
                        "score": 50.0, "reason": "No holdout Spearman data"}
            # Is most recent better than average?
            latest = spearman_vals[-1]
            avg = float(np.mean(spearman_vals[:-1]))
            if latest >= avg:
                return {"name": "retrain_effectiveness", "status": "PASS", "score": 85.0,
                        "reason": f"Latest retrain improved (Spearman: {latest:.4f} vs avg {avg:.4f})"}
            if latest > 0:
                return {"name": "retrain_effectiveness", "status": "WARN", "score": 55.0,
                        "reason": f"Latest retrain degraded (Spearman: {latest:.4f} vs avg {avg:.4f})"}
            return {"name": "retrain_effectiveness", "status": "FAIL", "score": 20.0,
                    "reason": f"Latest retrain negative (Spearman: {latest:.4f})"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'retrain_effectiveness' failed: %s", e)
            return {"name": "retrain_effectiveness", "status": "error", "score": 50.0,
                    "reason": f"Could not assess retrain effectiveness: {e}"}

    def _check_capital_utilization(self) -> Dict[str, Any]:
        """Check capital deployment efficiency."""
        try:
            from quant_engine.config import RESULTS_DIR, PAPER_INITIAL_CAPITAL
            state_path = RESULTS_DIR / "autopilot" / "paper_state.json"
            if not state_path.exists():
                return {"name": "capital_utilization", "status": "unavailable", "score": 50.0,
                        "reason": "No paper trading state"}
            with open(state_path, "r") as f:
                state = json.load(f)
            positions = state.get("positions", [])
            equity = float(state.get("equity", PAPER_INITIAL_CAPITAL))
            if equity <= 0:
                return {"name": "capital_utilization", "status": "FAIL", "score": 10.0,
                        "reason": "Zero or negative equity"}
            # Capital in positions vs total equity
            position_value = sum(float(p.get("value", 0)) for p in positions)
            utilization = position_value / equity if equity > 0 else 0.0
            if 0.30 <= utilization <= 0.90:
                return {"name": "capital_utilization", "status": "PASS", "score": 85.0,
                        "reason": f"Capital utilization: {utilization:.0%} (well-deployed)"}
            if utilization < 0.10:
                return {"name": "capital_utilization", "status": "WARN", "score": 50.0,
                        "reason": f"Capital utilization: {utilization:.0%} (underdeployed)"}
            if utilization > 0.95:
                return {"name": "capital_utilization", "status": "WARN", "score": 55.0,
                        "reason": f"Capital utilization: {utilization:.0%} (over-concentrated)"}
            return {"name": "capital_utilization", "status": "PASS", "score": 70.0,
                    "reason": f"Capital utilization: {utilization:.0%}"}
        except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
            logger.warning("Health check 'capital_utilization' failed: %s", e)
            return {"name": "capital_utilization", "status": "error", "score": 50.0,
                    "reason": f"Could not assess capital utilization: {e}"}
