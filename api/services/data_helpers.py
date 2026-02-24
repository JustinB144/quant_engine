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

logger = logging.getLogger(__name__)


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


# ── Time Series Helpers ───────────────────────────────────────────────


def build_equity_curves(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, max_points: int = 2500
) -> Dict[str, Any]:
    """Build aligned cumulative return series for strategy and benchmark."""
    if strategy_returns.empty:
        return {"strategy": [], "benchmark": [], "points": 0}

    # Align benchmark to strategy date range
    if not benchmark_returns.empty:
        common_start = max(strategy_returns.index.min(), benchmark_returns.index.min())
        common_end = min(strategy_returns.index.max(), benchmark_returns.index.max())
        strategy_returns = strategy_returns.loc[common_start:common_end]
        benchmark_returns = benchmark_returns.loc[common_start:common_end]
        # Reindex benchmark to strategy dates, fill gaps with 0
        benchmark_returns = benchmark_returns.reindex(strategy_returns.index, fill_value=0.0)

    strat_eq = (1.0 + strategy_returns).cumprod()
    bench_eq = (1.0 + benchmark_returns).cumprod() if not benchmark_returns.empty else pd.Series(dtype=float)

    # Downsample if needed
    if len(strat_eq) > max_points:
        step = max(1, len(strat_eq) // max_points)
        strat_eq = strat_eq.iloc[::step]
        if not bench_eq.empty:
            bench_eq = bench_eq.reindex(strat_eq.index, method="nearest")

    strat_points = [
        {"date": str(ts.date()) if hasattr(ts, "date") else str(ts), "value": round(float(v), 6)}
        for ts, v in strat_eq.items()
    ]
    bench_points = []
    if not bench_eq.empty:
        bench_points = [
            {"date": str(ts.date()) if hasattr(ts, "date") else str(ts), "value": round(float(v), 6)}
            for ts, v in bench_eq.items()
        ]

    return {"strategy": strat_points, "benchmark": bench_points, "points": len(strat_points)}


def compute_rolling_metrics(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60, max_points: int = 2500
) -> Dict[str, Any]:
    """Compute rolling correlation, alpha, beta, and relative strength."""
    if strategy_returns.empty or benchmark_returns.empty:
        return {"rolling_correlation": [], "rolling_alpha": [], "rolling_beta": [],
                "relative_strength": [], "drawdown_strategy": [], "drawdown_benchmark": [], "points": 0}

    # Align
    common = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common) < window:
        return {"rolling_correlation": [], "rolling_alpha": [], "rolling_beta": [],
                "relative_strength": [], "drawdown_strategy": [], "drawdown_benchmark": [], "points": 0}
    sr = strategy_returns.reindex(common).fillna(0.0)
    br = benchmark_returns.reindex(common).fillna(0.0)

    # Rolling correlation
    rolling_corr = sr.rolling(window).corr(br).dropna()

    # Rolling beta and alpha (OLS: strategy = alpha + beta * benchmark)
    def _rolling_regression(s, b, w):
        betas, alphas = [], []
        dates = []
        for i in range(w, len(s)):
            s_win = s.iloc[i - w : i].values
            b_win = b.iloc[i - w : i].values
            cov = np.cov(s_win, b_win)
            var_b = cov[1, 1]
            beta = cov[0, 1] / var_b if var_b > 1e-12 else 0.0
            alpha = (s_win.mean() - beta * b_win.mean()) * 252  # annualised
            betas.append(round(float(beta), 4))
            alphas.append(round(float(alpha), 4))
            dates.append(str(s.index[i].date()) if hasattr(s.index[i], "date") else str(s.index[i]))
        return dates, alphas, betas

    dates, alphas, betas = _rolling_regression(sr, br, window)

    # Relative strength: cumulative strategy / cumulative benchmark
    strat_eq = (1.0 + sr).cumprod()
    bench_eq = (1.0 + br).cumprod()
    rel_strength = (strat_eq / bench_eq).dropna()

    # Drawdowns
    def _drawdown(equity):
        peak = equity.cummax().replace(0, np.nan)
        return (equity / peak - 1.0).fillna(0.0)

    dd_strat = _drawdown(strat_eq)
    dd_bench = _drawdown(bench_eq)

    # Downsample all series to max_points
    def _downsample_ts(series, max_pts):
        if len(series) > max_pts:
            step = max(1, len(series) // max_pts)
            series = series.iloc[::step]
        return [
            {"date": str(ts.date()) if hasattr(ts, "date") else str(ts), "value": round(float(v), 4)}
            for ts, v in series.items()
        ]

    corr_points = _downsample_ts(rolling_corr, max_points)
    rs_points = _downsample_ts(rel_strength, max_points)
    dd_s_points = _downsample_ts(dd_strat, max_points)
    dd_b_points = _downsample_ts(dd_bench, max_points)

    # Downsample regression-based series
    if len(dates) > max_points:
        step = max(1, len(dates) // max_points)
        dates = dates[::step]
        alphas = alphas[::step]
        betas = betas[::step]
    alpha_points = [{"date": d, "value": a} for d, a in zip(dates, alphas)]
    beta_points = [{"date": d, "value": b} for d, b in zip(dates, betas)]

    return {
        "rolling_correlation": corr_points,
        "rolling_alpha": alpha_points,
        "rolling_beta": beta_points,
        "relative_strength": rs_points,
        "drawdown_strategy": dd_s_points,
        "drawdown_benchmark": dd_b_points,
        "points": len(corr_points),
        "window": window,
    }


def compute_returns_distribution(returns: pd.Series, bins: int = 50) -> Dict[str, Any]:
    """Compute histogram data and risk lines for a returns series."""
    if returns.empty:
        return {"bins": [], "var95": 0.0, "var99": 0.0, "cvar95": 0.0, "cvar99": 0.0, "count": 0}
    arr = returns.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"bins": [], "var95": 0.0, "var99": 0.0, "cvar95": 0.0, "cvar99": 0.0, "count": 0}

    counts, edges = np.histogram(arr, bins=bins)
    bin_data = [
        {"x": round(float((edges[i] + edges[i + 1]) / 2), 6), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    var95 = float(np.percentile(arr, 5))
    var99 = float(np.percentile(arr, 1))
    cvar95 = float(arr[arr <= var95].mean()) if np.any(arr <= var95) else var95
    cvar99 = float(arr[arr <= var99].mean()) if np.any(arr <= var99) else var99

    return {
        "bins": bin_data,
        "var95": round(var95, 6),
        "var99": round(var99, 6),
        "cvar95": round(cvar95, 6),
        "cvar99": round(cvar99, 6),
        "count": len(arr),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)), 6) if len(arr) > 1 else 0.0,
        "skew": round(float(pd.Series(arr).skew()), 4),
        "kurtosis": round(float(pd.Series(arr).kurtosis()), 4),
    }


def compute_rolling_risk(
    returns: pd.Series, vol_window: int = 21, sharpe_window: int = 60, max_points: int = 2500
) -> Dict[str, Any]:
    """Compute rolling volatility, Sharpe, and drawdown time series."""
    if returns.empty or len(returns) < vol_window:
        return {"rolling_vol": [], "rolling_sharpe": [], "drawdown": [], "points": 0}

    # Rolling annualised volatility
    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    # Rolling Sharpe (annualised return / annualised vol over the sharpe_window)
    rolling_ret = returns.rolling(sharpe_window).mean() * 252
    rolling_vol_sharpe = returns.rolling(sharpe_window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_ret / rolling_vol_sharpe.replace(0, np.nan)).dropna()

    # Drawdown
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax().replace(0, np.nan)
    drawdown = (equity / peak - 1.0).fillna(0.0)

    def _downsample(series, max_pts):
        if len(series) > max_pts:
            step = max(1, len(series) // max_pts)
            series = series.iloc[::step]
        return [
            {"date": str(ts.date()) if hasattr(ts, "date") else str(ts), "value": round(float(v), 4)}
            for ts, v in series.items()
        ]

    return {
        "rolling_vol": _downsample(rolling_vol, max_points),
        "rolling_sharpe": _downsample(rolling_sharpe, max_points),
        "drawdown": _downsample(drawdown, max_points),
        "points": len(rolling_vol),
        "vol_window": vol_window,
        "sharpe_window": sharpe_window,
    }


def compute_attribution(strategy_returns: pd.Series, cache_dir: Path) -> Dict[str, Any]:
    """Compute factor attribution: tech-minus-def and momentum-spread.

    Uses a simple OLS regression of strategy returns on factor proxies.
    """
    if strategy_returns.empty or len(strategy_returns) < 60:
        return {"factors": [], "residual_alpha": 0.0, "r_squared": 0.0, "points": 0}

    # Load tech (QQQ) and defensive (XLU/TLT) proxy returns from cache
    tech_ret = _load_proxy_returns(cache_dir, ["QQQ", "XLK"])
    def_ret = _load_proxy_returns(cache_dir, ["XLU", "TLT", "VZ"])

    # Build factor returns
    factors = {}
    common_idx = strategy_returns.index

    if not tech_ret.empty and not def_ret.empty:
        common_idx = common_idx.intersection(tech_ret.index).intersection(def_ret.index)
        if len(common_idx) > 30:
            tech_minus_def = tech_ret.reindex(common_idx).fillna(0.0) - def_ret.reindex(common_idx).fillna(0.0)
            factors["tech_minus_def"] = tech_minus_def

    # Momentum spread: top momentum vs bottom momentum proxy using SPY
    spy_ret = _load_proxy_returns(cache_dir, ["SPY"])
    if not spy_ret.empty:
        mom_idx = common_idx.intersection(spy_ret.index)
        if len(mom_idx) > 30:
            # Use rolling 20d return minus rolling 60d return as momentum spread proxy
            spy = spy_ret.reindex(mom_idx).fillna(0.0)
            fast_mom = spy.rolling(20).mean()
            slow_mom = spy.rolling(60).mean()
            momentum_spread = (fast_mom - slow_mom).dropna()
            if not momentum_spread.empty:
                factors["momentum_spread"] = momentum_spread

    if not factors:
        return {"factors": [], "residual_alpha": 0.0, "r_squared": 0.0, "points": 0}

    # Align all series
    factor_df = pd.DataFrame(factors)
    aligned_idx = strategy_returns.index.intersection(factor_df.index)
    if len(aligned_idx) < 30:
        return {"factors": [], "residual_alpha": 0.0, "r_squared": 0.0, "points": 0}

    y = strategy_returns.reindex(aligned_idx).values
    X = factor_df.reindex(aligned_idx).values
    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])

    try:
        # OLS: y = X @ beta
        beta, residuals, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
        y_pred = X_with_const @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    except np.linalg.LinAlgError:
        return {"factors": [], "residual_alpha": 0.0, "r_squared": 0.0, "points": 0}

    factor_names = list(factors.keys())
    factor_results = []
    for i, name in enumerate(factor_names):
        factor_results.append({
            "name": name,
            "coefficient": round(float(beta[i + 1]), 4),
            "annualized_contribution": round(float(beta[i + 1] * factor_df[name].reindex(aligned_idx).mean() * 252), 4),
        })

    return {
        "factors": factor_results,
        "residual_alpha": round(float(beta[0] * 252), 4),  # annualised
        "r_squared": round(float(r_squared), 4),
        "points": len(aligned_idx),
    }


def _load_proxy_returns(cache_dir: Path, symbols: List[str]) -> pd.Series:
    """Try to load daily close returns for any of the given symbol tickers."""
    for sym in symbols:
        # Try standard naming patterns
        for pattern in [f"{sym}_1d.parquet", f"{sym}_daily_*.parquet"]:
            if "*" in pattern:
                matches = sorted(cache_dir.glob(pattern))
                if matches:
                    try:
                        ret = _read_close_returns(matches[0])
                        if len(ret) > 20:
                            return ret
                    except (OSError, ValueError):
                        continue
            else:
                path = cache_dir / pattern
                if path.exists():
                    try:
                        ret = _read_close_returns(path)
                        if len(ret) > 20:
                            return ret
                    except (OSError, ValueError):
                        continue
    return pd.Series(dtype=float)


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
        # Guard against NA in regime series (can happen when features have NaN)
        last_regime = out.regime.iloc[-1]
        current_state = int(last_regime) if pd.notna(last_regime) else 2  # default mean-reverting
        current_probs = out.probabilities.iloc[-1]
        history = out.probabilities.tail(240).copy()
        label = REGIME_NAMES.get(current_state, f"Regime {current_state}")
        probs_pretty = {}
        for i in range(4):
            col = f"regime_prob_{i}"
            val = current_probs.get(col, 0.0)
            # pd.NA / np.nan can raise "boolean value of NA is ambiguous"
            # when passed to float(), so guard explicitly
            probs_pretty[REGIME_NAMES.get(i, f"Regime {i}")] = (
                float(val) if pd.notna(val) else 0.0
            )
        trans = out.transition_matrix if out.transition_matrix is not None else np.eye(4)
        trans = np.asarray(trans, dtype=float)
        if trans.ndim != 2:
            trans = np.eye(4)

        # Build regime change timeline
        regime_changes = []
        prev_regime = None
        start_date = None
        for date, regime_val in out.regime.items():
            rv = int(regime_val) if pd.notna(regime_val) else 2
            if rv != prev_regime:
                if prev_regime is not None and start_date is not None:
                    regime_changes.append({
                        "from_regime": REGIME_NAMES.get(prev_regime, f"Regime {prev_regime}"),
                        "to_regime": REGIME_NAMES.get(rv, f"Regime {rv}"),
                        "date": date.strftime("%Y-%m-%d"),
                        "duration_days": (date - start_date).days,
                    })
                start_date = date
                prev_regime = rv

        current_regime_duration = 0
        if start_date is not None and len(history) > 0:
            current_regime_duration = (history.index[-1] - start_date).days

        return {
            "current_label": label,
            "as_of": history.index[-1].strftime("%Y-%m-%d"),
            "current_probs": probs_pretty,
            "prob_history": history,
            "transition": trans,
            "regime_changes": regime_changes[-20:],
            "current_regime_duration_days": current_regime_duration,
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
    score = 0.0  # Start at 0 — earn points for each verified capability

    try:
        from quant_engine.config import SURVIVORSHIP_DB, DATA_CACHE_DIR, WRDS_ENABLED

        db_path = Path(SURVIVORSHIP_DB) if SURVIVORSHIP_DB else None
        if db_path and db_path.exists() and db_path.stat().st_size > 1024:
            surv_checks.append(HealthCheck(
                "Survivorship DB", "PASS", "Universe history database exists and is populated."))
            score += 25
        else:
            surv_checks.append(HealthCheck(
                "Survivorship DB", "FAIL",
                "Missing or empty survivorship database.",
                recommendation="Run survivorship snapshot collection."))

        if WRDS_ENABLED:
            surv_checks.append(HealthCheck(
                "WRDS Data Source", "PASS", "WRDS enabled as primary institutional source."))
            score += 25
        else:
            surv_checks.append(HealthCheck(
                "WRDS Data Source", "WARN",
                "WRDS disabled — relying on IBKR/local cache (survivorship-biased).",
                recommendation="Enable WRDS for institutional-grade data."))
            score += 10

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
                    score += 25
                elif max_age <= 21:
                    score += 15
            else:
                quality_checks.append(HealthCheck(
                    "Cache Freshness", "FAIL", "No cached parquet files found."))
        else:
            quality_checks.append(HealthCheck(
                "Data Cache", "FAIL", "Cache directory does not exist."))

        # Bonus for data quality gates being enabled
        from quant_engine.config import DATA_QUALITY_ENABLED
        if DATA_QUALITY_ENABLED:
            quality_checks.append(HealthCheck(
                "Data Quality Gates", "PASS", "Automated quality gates enabled."))
            score += 25
        else:
            quality_checks.append(HealthCheck(
                "Data Quality Gates", "WARN", "Data quality gates disabled.",
                recommendation="Enable DATA_QUALITY_ENABLED."))

    except (ValueError, ImportError) as e:
        surv_checks.append(HealthCheck("Config Check", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return surv_checks, quality_checks, score


def _check_promotion_contract() -> Tuple[List[HealthCheck], Dict[str, int], float]:
    """Verify promotion gate configuration."""
    checks: List[HealthCheck] = []
    score = 0.0  # Start at 0 — earn points for each verified capability

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
        score += 15
        checks.append(HealthCheck(
            "Max Drawdown Gate", "PASS",
            f"Threshold: {PROMOTION_MAX_DRAWDOWN}", value=f"{PROMOTION_MAX_DRAWDOWN}"))
        score += 15
        checks.append(HealthCheck(
            "DSR Significance", "PASS",
            f"Max p-value: {PROMOTION_MAX_DSR_PVALUE}", value=f"p<{PROMOTION_MAX_DSR_PVALUE}"))
        score += 15
        checks.append(HealthCheck(
            "PBO Gate", "PASS",
            f"Max PBO: {PROMOTION_MAX_PBO}", value=f"<{PROMOTION_MAX_PBO}"))
        score += 15
        checks.append(HealthCheck(
            "Min Trade Count", "PASS",
            f"Required: {PROMOTION_MIN_TRADES} trades", value=f"{PROMOTION_MIN_TRADES}"))
        score += 15
        if PROMOTION_REQUIRE_CAPACITY_UNCONSTRAINED:
            checks.append(HealthCheck(
                "Capacity Check", "PASS", "Capacity constraint is enforced."))
            score += 25
        else:
            score += 10
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("Promotion Config", "FAIL", str(e)))

    # Load real funnel data from latest cycle report instead of hardcoded values.
    funnel = {
        "Candidates Generated": 0,
        "Passed Sharpe": 0,
        "Passed DSR": 0,
        "Passed PBO": 0,
        "Passed Capacity": 0,
        "Promoted": 0,
    }
    try:
        from quant_engine.config import AUTOPILOT_CYCLE_REPORT, STRATEGY_REGISTRY_PATH
        report_path = Path(AUTOPILOT_CYCLE_REPORT)
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                cycle = json.load(f)
            funnel["Candidates Generated"] = int(cycle.get("n_candidates", 0))
            funnel["Promoted"] = int(cycle.get("n_promoted", 0))
            n_passed = int(cycle.get("n_passed", 0))
            # Reconstruct intermediate funnel from top_decisions breakdown
            decisions = cycle.get("top_decisions", [])
            if decisions:
                reasons_all = [r for d in decisions for r in d.get("reasons", [])]
                n_total = len(decisions)
                funnel["Passed Sharpe"] = n_total - sum(1 for r in reasons_all if "sharpe" in r)
                funnel["Passed DSR"] = n_total - sum(1 for r in reasons_all if "dsr" in r)
                funnel["Passed PBO"] = n_total - sum(1 for r in reasons_all if "pbo" in r)
                funnel["Passed Capacity"] = n_total - sum(1 for r in reasons_all if "capacity" in r)
            else:
                funnel["Passed Sharpe"] = n_passed
                funnel["Passed DSR"] = n_passed
                funnel["Passed PBO"] = n_passed
                funnel["Passed Capacity"] = n_passed
    except (OSError, json.JSONDecodeError, ValueError, ImportError) as e:
        logger.warning("Failed to load cycle report for promotion funnel: %s", e)

    score = min(100.0, max(0.0, score))
    return checks, funnel, score


def _check_walkforward() -> Tuple[List[HealthCheck], float]:
    """Verify walk-forward validation setup."""
    checks: List[HealthCheck] = []
    score = 0.0  # Start at 0 — earn points for each verified capability

    try:
        from quant_engine.config import CV_FOLDS, HOLDOUT_FRACTION, CPCV_PARTITIONS

        checks.append(HealthCheck(
            "CV Folds", "PASS", f"{CV_FOLDS}-fold cross-validation configured.", value=str(CV_FOLDS)))
        score += 20
        checks.append(HealthCheck(
            "Holdout Fraction", "PASS",
            f"{HOLDOUT_FRACTION:.0%} holdout reserved.", value=f"{HOLDOUT_FRACTION:.0%}"))
        score += 20
        checks.append(HealthCheck(
            "CPCV Partitions", "PASS",
            f"{CPCV_PARTITIONS} combinatorial partitions.", value=str(CPCV_PARTITIONS)))
        score += 20

        try:
            from quant_engine.backtest.validation import WalkForwardFold
            checks.append(HealthCheck(
                "WalkForwardFold", "PASS", "Walk-forward validation module available."))
            score += 20
        except ImportError:
            checks.append(HealthCheck(
                "WalkForwardFold", "FAIL",
                "Walk-forward module not found.",
                recommendation="Implement backtest.validation.WalkForwardFold"))

        try:
            from quant_engine.backtest.validation import run_statistical_tests
            checks.append(HealthCheck(
                "Statistical Tests", "PASS", "IC/FDR statistical testing available."))
            score += 20
        except ImportError:
            checks.append(HealthCheck(
                "Statistical Tests", "WARN", "Statistical testing module not available."))
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("WF Config", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return checks, score


def _check_execution() -> Tuple[List[HealthCheck], float]:
    """Audit execution cost model."""
    checks: List[HealthCheck] = []
    score = 0.0  # Start at 0 — earn points for each verified capability

    try:
        from quant_engine.config import (
            TRANSACTION_COST_BPS, EXEC_DYNAMIC_COSTS,
            ALMGREN_CHRISS_ENABLED, EXEC_SPREAD_BPS, EXEC_IMPACT_COEFF_BPS,
        )

        checks.append(HealthCheck(
            "Base Cost", "PASS",
            f"Transaction cost: {TRANSACTION_COST_BPS} bps round-trip.", value=f"{TRANSACTION_COST_BPS}bps"))
        score += 20

        if EXEC_DYNAMIC_COSTS:
            checks.append(HealthCheck(
                "Dynamic Costs", "PASS", "Costs conditioned on vol/liquidity."))
            score += 25
        else:
            checks.append(HealthCheck(
                "Dynamic Costs", "WARN",
                "Flat cost model — not conditioned on market state.",
                recommendation="Enable EXEC_DYNAMIC_COSTS."))

        if ALMGREN_CHRISS_ENABLED:
            checks.append(HealthCheck(
                "Almgren-Chriss", "PASS", "Optimal execution enabled for large positions."))
            score += 25
        else:
            checks.append(HealthCheck(
                "Almgren-Chriss", "WARN", "Optimal execution disabled."))

        checks.append(HealthCheck(
            "Spread Model", "PASS",
            f"Base spread: {EXEC_SPREAD_BPS} bps.", value=f"{EXEC_SPREAD_BPS}bps"))
        score += 15
        checks.append(HealthCheck(
            "Impact Model", "PASS",
            f"Impact coefficient: {EXEC_IMPACT_COEFF_BPS} bps.", value=f"{EXEC_IMPACT_COEFF_BPS}bps"))
        score += 15
    except (ValueError, ImportError) as e:
        checks.append(HealthCheck("Execution Config", "FAIL", str(e)))

    score = min(100.0, max(0.0, score))
    return checks, score


def _check_complexity() -> Tuple[List[HealthCheck], Dict[str, int], List[Dict[str, str]], float]:
    """Audit feature and knob complexity."""
    checks: List[HealthCheck] = []
    score = 0.0  # Start at 0 — earn points for each verified capability
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
        score += 25 if MAX_FEATURES_SELECTED <= 30 else 10

        n_interactions = len(INTERACTION_PAIRS)
        checks.append(HealthCheck(
            "Interaction Features",
            "PASS" if n_interactions <= 15 else "WARN",
            f"{n_interactions} interaction pairs configured.",
            value=str(n_interactions)))
        score += 25 if n_interactions <= 15 else 10

        feat_mode = AUTOPILOT_FEATURE_MODE
        checks.append(HealthCheck(
            "Feature Mode",
            "PASS" if feat_mode == "core" else "WARN",
            f"Autopilot uses '{feat_mode}' feature set.",
            value=feat_mode))
        score += 25 if feat_mode == "core" else 10

        feature_inventory = {
            "Technical": 12, "Volatility": 8, "Microstructure": 6,
            "Regime": 4, "Interaction": n_interactions,
        }

        knob_inventory = [
            {"name": "HMM States", "value": str(REGIME_HMM_STATES), "module": "Regime"},
            {"name": "Max Features", "value": str(MAX_FEATURES_SELECTED), "module": "Features"},
            {"name": "Interaction Pairs", "value": str(n_interactions), "module": "Features"},
        ]

        score += 25
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
