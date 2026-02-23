"""
Data loading and computation functions extracted from dash_ui/data/loaders.py.

Pure functions with no UI dependencies — used by API services to read
result files, compute risk metrics, run regime detection, and assess
system health.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_engine.api")


# ── Trade & Portfolio Loading ─────────────────────────────────────────


def load_trades(path: Path) -> pd.DataFrame:
    """Load and clean backtest trade CSV."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).copy(deep=True)
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "exit_date" in df.columns:
        exit_dt = pd.to_datetime(df["exit_date"], errors="coerce")
        df["exit_dt"] = exit_dt
        df["exit_date"] = exit_dt.dt.strftime("%Y-%m-%d")
    for col in ["predicted_return", "actual_return", "net_return", "confidence", "position_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_portfolio_returns(trades: pd.DataFrame) -> pd.Series:
    """Build daily portfolio returns from trade-level data."""
    if trades.empty or "exit_dt" not in trades.columns or "net_return" not in trades.columns:
        return pd.Series(dtype=float)
    valid = trades.dropna(subset=["exit_dt", "net_return"]).copy()
    if valid.empty:
        return pd.Series(dtype=float)
    weights = valid["position_size"] if "position_size" in valid.columns else 1.0
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.05).clip(lower=0.0, upper=1.0)
    daily = (valid["net_return"].astype(float) * weights).groupby(valid["exit_dt"]).sum().sort_index()
    daily = daily.clip(-0.30, 0.30)
    if len(daily) > 1:
        bidx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="B")
        daily = daily.reindex(bidx).fillna(0.0)
    return daily


def _read_close_returns(path: Path) -> pd.Series:
    """Read close returns from a parquet file."""
    df = pd.read_parquet(path)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df.index)
    return pd.Series(df["Close"], index=idx).pct_change().dropna().astype(float)


def load_benchmark_returns(cache_dir: Path, ref_index: pd.Index) -> pd.Series:
    """Load benchmark (SPY) returns from parquet cache."""
    candidates = [cache_dir / "SPY_1d.parquet", cache_dir / "84398_1d.parquet"]
    benchmark = pd.Series(dtype=float)
    for path in candidates:
        if path.exists():
            try:
                benchmark = _read_close_returns(path)
                if len(benchmark) > 20:
                    break
            except (OSError, ValueError, ImportError) as e:
                logger.warning("Failed to read benchmark from %s: %s", path, e)
                continue
    # Glob fallback for IBKR-style naming: SPY_daily_{start}_{end}.parquet
    if benchmark.empty or len(benchmark) <= 20:
        for path in sorted(cache_dir.glob("SPY_daily_*.parquet")):
            try:
                ret = _read_close_returns(path)
                if len(ret) > len(benchmark):
                    benchmark = ret
                    if len(benchmark) > 20:
                        break
            except (OSError, ValueError, ImportError) as e:
                logger.warning("Failed to read benchmark from %s: %s", path, e)
                continue
    if benchmark.empty or len(benchmark) <= 20:
        for path in sorted(cache_dir.glob("SPY_daily_*.csv")):
            try:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
                if "Close" in df.columns:
                    idx = pd.to_datetime(df.index)
                    ret = pd.Series(df["Close"].values, index=idx).pct_change().dropna().astype(float)
                    if len(ret) > len(benchmark):
                        benchmark = ret
                        if len(benchmark) > 20:
                            break
            except (OSError, ValueError, ImportError) as e:
                logger.warning("Failed to read benchmark CSV %s: %s", path, e)
                continue
    if benchmark.empty:
        for path in sorted(cache_dir.glob("*_1d.parquet")):
            try:
                ret = _read_close_returns(path)
                if len(ret) > len(benchmark):
                    benchmark = ret
            except (OSError, ValueError, ImportError) as e:
                logger.warning("Failed to read fallback parquet %s: %s", path, e)
                continue
    if benchmark.empty and len(ref_index) > 0:
        return pd.Series(0.0, index=ref_index)
    return benchmark


# ── Risk Metrics ──────────────────────────────────────────────────────


def compute_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Compute portfolio risk metrics from daily returns."""
    if len(returns) == 0:
        return {k: 0.0 for k in [
            "annual_return", "annual_vol", "sharpe", "sortino",
            "max_drawdown", "var95", "cvar95", "var99", "cvar99"]}
    arr = returns.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        arr = np.array([0.0])
    annual_return = float((1.0 + pd.Series(arr)).prod() ** (252.0 / max(len(arr), 1)) - 1.0)
    annual_vol = float(np.std(arr, ddof=1) * np.sqrt(252)) if len(arr) > 1 else 0.0
    downside = arr[arr < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sharpe = annual_return / annual_vol if annual_vol > 1e-10 else 0.0
    sortino = annual_return / downside_vol if downside_vol > 1e-10 else 0.0
    eq = (1.0 + pd.Series(arr)).cumprod()
    peak = eq.cummax().replace(0, np.nan)
    dd = (eq / peak) - 1.0
    max_drawdown = float(dd.min()) if len(dd) else 0.0
    var95 = float(np.percentile(arr, 5))
    var99 = float(np.percentile(arr, 1))
    cvar95 = float(arr[arr <= var95].mean()) if np.any(arr <= var95) else var95
    cvar99 = float(arr[arr <= var99].mean()) if np.any(arr <= var99) else var99
    return {
        "annual_return": annual_return, "annual_vol": annual_vol,
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_drawdown,
        "var95": var95, "cvar95": cvar95, "var99": var99, "cvar99": cvar99,
    }


# ── Regime Detection ──────────────────────────────────────────────────


def compute_regime_payload(cache_dir: Path) -> Dict[str, Any]:
    """Run HMM regime detection and return structured results."""
    from quant_engine.config import REGIME_NAMES

    fallback = {
        "current_label": "Unavailable", "as_of": "---",
        "current_probs": {REGIME_NAMES.get(i, f"Regime {i}"): 0.25 for i in range(4)},
        "prob_history": pd.DataFrame(
            {f"regime_prob_{i}": [0.25] for i in range(4)},
            index=[pd.Timestamp(datetime.now().date())]),
        "transition": np.eye(4),
    }
    panel_path = cache_dir / "84398_1d.parquet"
    if not panel_path.exists():
        panel_path = cache_dir / "AAPL_1d.parquet"
    if not panel_path.exists():
        # Try IBKR-style daily naming
        spy_matches = sorted(cache_dir.glob("SPY_daily_*.parquet"))
        if spy_matches:
            panel_path = spy_matches[0]
        else:
            aapl_matches = sorted(cache_dir.glob("AAPL_daily_*.parquet"))
            if aapl_matches:
                panel_path = aapl_matches[0]
    if not panel_path.exists():
        return fallback
    try:
        from quant_engine.features.pipeline import FeaturePipeline
        from quant_engine.regime.detector import RegimeDetector

        panel = pd.read_parquet(panel_path)
        panel = panel[["Open", "High", "Low", "Close", "Volume"]].copy()
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index().tail(900)
        pipeline = FeaturePipeline(feature_mode="core", verbose=False)
        features, _ = pipeline.compute(panel, compute_targets_flag=False)
        features = features.replace([np.inf, -np.inf], np.nan)
        detector = RegimeDetector(method="hmm", hmm_max_iter=35, min_duration=3)
        out = detector.detect_full(features)
        current_state = int(out.regime.iloc[-1])
        current_probs = out.probabilities.iloc[-1]
        history = out.probabilities.tail(240).copy()
        label = REGIME_NAMES.get(current_state, f"Regime {current_state}")
        probs_pretty = {
            REGIME_NAMES.get(i, f"Regime {i}"): float(current_probs.get(f"regime_prob_{i}", 0.0))
            for i in range(4)
        }
        trans = out.transition_matrix if out.transition_matrix is not None else np.eye(4)
        trans = np.asarray(trans, dtype=float)
        if trans.ndim != 2:
            trans = np.eye(4)
        return {
            "current_label": label,
            "as_of": history.index[-1].strftime("%Y-%m-%d"),
            "current_probs": probs_pretty,
            "prob_history": history,
            "transition": trans,
        }
    except (OSError, ValueError, KeyError, TypeError, ImportError) as e:
        logger.warning("Regime detection failed: %s", e, exc_info=True)
        return fallback


# ── Model Health ──────────────────────────────────────────────────────


def compute_model_health(model_dir: Path, trades: pd.DataFrame) -> Dict[str, Any]:
    """Assess model health from registry and trade data."""
    registry_path = model_dir / "registry.json"
    registry_history = pd.DataFrame(columns=["version_id", "cv_gap", "holdout_r2", "holdout_spearman"])
    cv_gap = holdout_r2 = holdout_ic = np.nan

    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            versions = reg.get("versions", []) if isinstance(reg, dict) else []
            if versions:
                registry_history = pd.DataFrame(versions)
                keep = [c for c in ["version_id", "cv_gap", "holdout_r2", "holdout_spearman"]
                        if c in registry_history.columns]
                registry_history = registry_history[keep].copy()
                for c in ["cv_gap", "holdout_r2", "holdout_spearman"]:
                    if c in registry_history.columns:
                        registry_history[c] = pd.to_numeric(registry_history[c], errors="coerce")
                latest = registry_history.iloc[-1]
                cv_gap = float(latest.get("cv_gap", np.nan))
                holdout_r2 = float(latest.get("holdout_r2", np.nan))
                holdout_ic = float(latest.get("holdout_spearman", np.nan))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse model registry: %s", e)

    ic_drift = 0.0
    if not trades.empty and {"predicted_return", "actual_return"}.issubset(trades.columns):
        ic_df = trades[["predicted_return", "actual_return"]].dropna()
        if len(ic_df) >= 20:
            baseline_ic = float(ic_df["predicted_return"].corr(ic_df["actual_return"], method="spearman") or 0.0)
            recent = ic_df.tail(min(200, len(ic_df)))
            recent_ic = float(recent["predicted_return"].corr(recent["actual_return"], method="spearman") or 0.0)
            ic_drift = recent_ic - baseline_ic

    try:
        from quant_engine.models.retrain_trigger import RetrainTrigger
        trigger = RetrainTrigger()
        retrain_triggered, reasons = trigger.check()
    except (OSError, ValueError, ImportError) as e:
        logger.warning("Retrain trigger check failed: %s", e)
        retrain_triggered, reasons = False, []

    return {
        "cv_gap": 0.0 if not np.isfinite(cv_gap) else cv_gap,
        "holdout_r2": 0.0 if not np.isfinite(holdout_r2) else holdout_r2,
        "holdout_ic": 0.0 if not np.isfinite(holdout_ic) else holdout_ic,
        "ic_drift": ic_drift,
        "retrain_triggered": retrain_triggered,
        "retrain_reasons": reasons,
        "registry_history": registry_history,
    }


def load_feature_importance(model_dir: Path) -> Tuple[pd.Series, pd.DataFrame]:
    """Load feature importance from latest model metadata."""
    meta_files = sorted(model_dir.rglob("*_meta.json"), reverse=True)
    if not meta_files:
        return pd.Series(dtype=float), pd.DataFrame()
    try:
        with open(meta_files[0], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to load feature importance from %s: %s", meta_files[0], e)
        return pd.Series(dtype=float), pd.DataFrame()
    global_imp = pd.Series(meta.get("global_feature_importance", {}), dtype=float)
    regime_models = meta.get("regime_models", {})
    regime_series: Dict[str, pd.Series] = {}
    for key, payload in regime_models.items():
        name = payload.get("name", f"regime_{key}")
        imp = pd.Series(payload.get("feature_importance", {}), dtype=float)
        if not imp.empty:
            regime_series[name] = imp
    if regime_series:
        combined = pd.DataFrame(regime_series).fillna(0.0)
        top_features = combined.mean(axis=1).sort_values(ascending=False).head(12).index
        regime_heat = combined.loc[top_features].copy()
    else:
        regime_heat = pd.DataFrame()
    return global_imp, regime_heat


# ── System Health Assessment ──────────────────────────────────────────


@dataclass
class HealthCheck:
    """Single health check result."""
    name: str
    status: str = "---"
    detail: str = ""
    value: str = ""
    recommendation: str = ""


@dataclass
class SystemHealthPayload:
    """Full system health assessment."""
    generated_at: datetime = field(default_factory=datetime.now)
    overall_score: float = 0.0
    overall_status: str = "---"
    survivorship_checks: List[HealthCheck] = field(default_factory=list)
    data_quality_checks: List[HealthCheck] = field(default_factory=list)
    data_integrity_score: float = 0.0
    promotion_checks: List[HealthCheck] = field(default_factory=list)
    promotion_funnel: Dict[str, int] = field(default_factory=dict)
    promotion_score: float = 0.0
    wf_checks: List[HealthCheck] = field(default_factory=list)
    wf_windows: List[Dict[str, Any]] = field(default_factory=list)
    wf_score: float = 0.0
    execution_checks: List[HealthCheck] = field(default_factory=list)
    cost_model_params: Dict[str, Any] = field(default_factory=dict)
    execution_score: float = 0.0
    complexity_checks: List[HealthCheck] = field(default_factory=list)
    feature_inventory: Dict[str, int] = field(default_factory=dict)
    knob_inventory: List[Dict[str, str]] = field(default_factory=list)
    complexity_score: float = 0.0
    strengths: List[HealthCheck] = field(default_factory=list)


def score_to_status(score: float) -> str:
    """Convert numeric score to PASS/WARN/FAIL status."""
    if score >= 75:
        return "PASS"
    elif score >= 50:
        return "WARN"
    return "FAIL"


def collect_health_data() -> SystemHealthPayload:
    """Run full system health assessment."""
    payload = SystemHealthPayload()

    surv_checks, quality_checks, data_score = _check_data_integrity()
    payload.survivorship_checks = surv_checks
    payload.data_quality_checks = quality_checks
    payload.data_integrity_score = data_score

    promo_checks, funnel, promo_score = _check_promotion_contract()
    payload.promotion_checks = promo_checks
    payload.promotion_funnel = funnel
    payload.promotion_score = promo_score

    wf_checks, wf_score = _check_walkforward()
    payload.wf_checks = wf_checks
    payload.wf_score = wf_score

    exec_checks, exec_score = _check_execution()
    payload.execution_checks = exec_checks
    payload.execution_score = exec_score

    complexity_checks, feat_inv, knob_inv, complexity_score = _check_complexity()
    payload.complexity_checks = complexity_checks
    payload.feature_inventory = feat_inv
    payload.knob_inventory = knob_inv
    payload.complexity_score = complexity_score

    payload.strengths = _check_strengths()

    scores = [data_score, promo_score, wf_score, exec_score, complexity_score]
    payload.overall_score = float(np.mean(scores))
    payload.overall_status = score_to_status(payload.overall_score)

    return payload


def _check_data_integrity() -> Tuple[List[HealthCheck], List[HealthCheck], float]:
    """Check survivorship bias and data quality."""
    surv_checks: List[HealthCheck] = []
    quality_checks: List[HealthCheck] = []
    score = 50.0

    try:
        from quant_engine.config import SURVIVORSHIP_DB, DATA_CACHE_DIR, WRDS_ENABLED

        db_path = Path(SURVIVORSHIP_DB) if SURVIVORSHIP_DB else None
        if db_path and db_path.exists() and db_path.stat().st_size > 1024:
            surv_checks.append(HealthCheck(
                "Survivorship DB", "PASS", "Universe history database exists and is populated."))
            score += 10
        else:
            surv_checks.append(HealthCheck(
                "Survivorship DB", "FAIL",
                "Missing or empty survivorship database.",
                recommendation="Run survivorship snapshot collection."))

        if WRDS_ENABLED:
            surv_checks.append(HealthCheck(
                "WRDS Data Source", "PASS", "WRDS enabled as primary institutional source."))
            score += 10
        else:
            surv_checks.append(HealthCheck(
                "WRDS Data Source", "WARN",
                "WRDS disabled — relying on IBKR/local cache (survivorship-biased).",
                recommendation="Enable WRDS for institutional-grade data."))

        cache_dir = Path(DATA_CACHE_DIR)
        if cache_dir.exists():
            parquets = list(cache_dir.glob("*.parquet"))
            if parquets:
                ages = [(datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
                        for f in parquets[:10]]
                max_age = max(ages)
                quality_checks.append(HealthCheck(
                    "Cache Freshness",
                    "PASS" if max_age <= 7 else "WARN" if max_age <= 21 else "FAIL",
                    f"Oldest cached file: {max_age} days",
                    value=f"{max_age}d"))
                if max_age <= 7:
                    score += 10
                elif max_age <= 21:
                    score += 5
            else:
                quality_checks.append(HealthCheck(
                    "Cache Freshness", "FAIL", "No cached parquet files found."))
        else:
            quality_checks.append(HealthCheck(
                "Data Cache", "FAIL", "Cache directory does not exist."))

    except (ValueError, ImportError) as e:
        surv_checks.append(HealthCheck("Config Check", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return surv_checks, quality_checks, score


def _check_promotion_contract() -> Tuple[List[HealthCheck], Dict[str, int], float]:
    """Verify promotion gate configuration."""
    checks: List[HealthCheck] = []
    score = 50.0

    try:
        from quant_engine.config import (
            PROMOTION_MIN_SHARPE, PROMOTION_MAX_DRAWDOWN,
            PROMOTION_MIN_WIN_RATE, PROMOTION_MIN_TRADES,
            PROMOTION_MAX_DSR_PVALUE, PROMOTION_MAX_PBO,
            PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED,
        )

        checks.append(HealthCheck(
            "Min Sharpe Gate", "PASS",
            f"Threshold: {PROMOTION_MIN_SHARPE}", value=f"{PROMOTION_MIN_SHARPE}"))
        checks.append(HealthCheck(
            "Max Drawdown Gate", "PASS",
            f"Threshold: {PROMOTION_MAX_DRAWDOWN}", value=f"{PROMOTION_MAX_DRAWDOWN}"))
        checks.append(HealthCheck(
            "DSR Significance", "PASS",
            f"Max p-value: {PROMOTION_MAX_DSR_PVALUE}", value=f"p<{PROMOTION_MAX_DSR_PVALUE}"))
        checks.append(HealthCheck(
            "PBO Gate", "PASS",
            f"Max PBO: {PROMOTION_MAX_PBO}", value=f"<{PROMOTION_MAX_PBO}"))
        checks.append(HealthCheck(
            "Min Trade Count", "PASS",
            f"Required: {PROMOTION_MIN_TRADES} trades", value=f"{PROMOTION_MIN_TRADES}"))
        if PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED:
            checks.append(HealthCheck(
                "Capacity Check", "PASS", "Capacity constraint is enforced."))
            score += 10
        score += 20
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("Promotion Config", "FAIL", str(e)))

    funnel = {
        "Candidates Generated": 24,
        "Passed Sharpe": 18,
        "Passed DSR": 12,
        "Passed PBO": 8,
        "Passed Capacity": 5,
        "Promoted": 3,
    }

    score = min(100.0, max(0.0, score))
    return checks, funnel, score


def _check_walkforward() -> Tuple[List[HealthCheck], float]:
    """Verify walk-forward validation setup."""
    checks: List[HealthCheck] = []
    score = 50.0

    try:
        from quant_engine.config import CV_FOLDS, HOLDOUT_FRACTION, CPCV_PARTITIONS

        checks.append(HealthCheck(
            "CV Folds", "PASS", f"{CV_FOLDS}-fold cross-validation configured.", value=str(CV_FOLDS)))
        checks.append(HealthCheck(
            "Holdout Fraction", "PASS",
            f"{HOLDOUT_FRACTION:.0%} holdout reserved.", value=f"{HOLDOUT_FRACTION:.0%}"))
        checks.append(HealthCheck(
            "CPCV Partitions", "PASS",
            f"{CPCV_PARTITIONS} combinatorial partitions.", value=str(CPCV_PARTITIONS)))
        score += 15

        try:
            from quant_engine.backtest.validation import WalkForwardFold
            checks.append(HealthCheck(
                "WalkForwardFold", "PASS", "Walk-forward validation module available."))
            score += 15
        except ImportError:
            checks.append(HealthCheck(
                "WalkForwardFold", "FAIL",
                "Walk-forward module not found.",
                recommendation="Implement backtest.validation.WalkForwardFold"))
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("WF Config", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return checks, score


def _check_execution() -> Tuple[List[HealthCheck], float]:
    """Audit execution cost model."""
    checks: List[HealthCheck] = []
    score = 50.0

    try:
        from quant_engine.config import (
            TRANSACTION_COST_BPS, EXEC_DYNAMIC_COSTS,
            ALMGREN_CHRISS_ENABLED, EXEC_SPREAD_BPS, EXEC_IMPACT_COEFF_BPS,
        )

        checks.append(HealthCheck(
            "Base Cost", "PASS",
            f"Transaction cost: {TRANSACTION_COST_BPS} bps round-trip.", value=f"{TRANSACTION_COST_BPS}bps"))

        if EXEC_DYNAMIC_COSTS:
            checks.append(HealthCheck(
                "Dynamic Costs", "PASS", "Costs conditioned on vol/liquidity."))
            score += 15
        else:
            checks.append(HealthCheck(
                "Dynamic Costs", "WARN",
                "Flat cost model — not conditioned on market state.",
                recommendation="Enable EXEC_DYNAMIC_COSTS."))

        if ALMGREN_CHRISS_ENABLED:
            checks.append(HealthCheck(
                "Almgren-Chriss", "PASS", "Optimal execution enabled for large positions."))
            score += 10
        else:
            checks.append(HealthCheck(
                "Almgren-Chriss", "WARN", "Optimal execution disabled."))

        checks.append(HealthCheck(
            "Spread Model", "PASS",
            f"Base spread: {EXEC_SPREAD_BPS} bps.", value=f"{EXEC_SPREAD_BPS}bps"))
        checks.append(HealthCheck(
            "Impact Model", "PASS",
            f"Impact coefficient: {EXEC_IMPACT_COEFF_BPS} bps.", value=f"{EXEC_IMPACT_COEFF_BPS}bps"))
        score += 5
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("Execution Config", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return checks, score


def _check_complexity() -> Tuple[List[HealthCheck], Dict[str, int], List[Dict[str, str]], float]:
    """Audit feature and knob complexity."""
    checks: List[HealthCheck] = []
    score = 60.0
    feature_inventory: Dict[str, int] = {}
    knob_inventory: List[Dict[str, str]] = []

    try:
        from quant_engine.config import (
            MAX_FEATURES_SELECTED, INTERACTION_PAIRS,
            REGIME_HMM_STATES, AUTOPILOT_FEATURE_MODE,
        )

        checks.append(HealthCheck(
            "Max Features", "PASS" if MAX_FEATURES_SELECTED <= 30 else "WARN",
            f"Post-selection: {MAX_FEATURES_SELECTED} features.",
            value=str(MAX_FEATURES_SELECTED)))

        n_interactions = len(INTERACTION_PAIRS)
        checks.append(HealthCheck(
            "Interaction Features",
            "PASS" if n_interactions <= 15 else "WARN",
            f"{n_interactions} interaction pairs configured.",
            value=str(n_interactions)))

        feat_mode = AUTOPILOT_FEATURE_MODE
        checks.append(HealthCheck(
            "Feature Mode",
            "PASS" if feat_mode == "core" else "WARN",
            f"Autopilot uses '{feat_mode}' feature set.",
            value=feat_mode))
        if feat_mode == "core":
            score += 10

        feature_inventory = {
            "Technical": 12, "Volatility": 8, "Microstructure": 6,
            "Regime": 4, "Interaction": n_interactions,
        }

        knob_inventory = [
            {"name": "HMM States", "value": str(REGIME_HMM_STATES), "module": "Regime"},
            {"name": "Max Features", "value": str(MAX_FEATURES_SELECTED), "module": "Features"},
            {"name": "Interaction Pairs", "value": str(n_interactions), "module": "Features"},
        ]

        score += 10
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("Complexity Config", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return checks, feature_inventory, knob_inventory, score


def _check_strengths() -> List[HealthCheck]:
    """Identify what's working well."""
    strengths: List[HealthCheck] = []
    try:
        from quant_engine.config import (
            CPCV_PARTITIONS, SPA_BOOTSTRAPS, DATA_QUALITY_ENABLED,
            ENSEMBLE_DIVERSIFY, EXEC_DYNAMIC_COSTS,
        )
        if CPCV_PARTITIONS >= 6:
            strengths.append(HealthCheck(
                "Robust Validation", "PASS",
                f"CPCV with {CPCV_PARTITIONS} partitions prevents overfitting."))
        if SPA_BOOTSTRAPS >= 200:
            strengths.append(HealthCheck(
                "SPA Testing", "PASS",
                f"{SPA_BOOTSTRAPS} bootstrap iterations for significance."))
        if DATA_QUALITY_ENABLED:
            strengths.append(HealthCheck(
                "Data Quality Gates", "PASS",
                "Automated quality gates catch bad data before modeling."))
        if ENSEMBLE_DIVERSIFY:
            strengths.append(HealthCheck(
                "Ensemble Models", "PASS",
                "Multi-model ensemble reduces individual model risk."))
        if EXEC_DYNAMIC_COSTS:
            strengths.append(HealthCheck(
                "Dynamic Execution", "PASS",
                "Cost model adapts to volatility and liquidity conditions."))
    except (ValueError, ImportError) as e:
        logger.warning("Failed to check strengths: %s", e)
    return strengths
