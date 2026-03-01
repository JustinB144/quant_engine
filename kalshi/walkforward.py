"""
Walk-forward evaluation for event-centric Kalshi feature panels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..backtest.advanced_validation import deflated_sharpe_ratio, monte_carlo_validation


@dataclass
class EventWalkForwardConfig:
    """Configuration for event-level walk-forward splits, purge/embargo, and trial accounting."""
    train_min_events: int = 40
    test_events_per_fold: int = 20
    step_events: int = 20
    purge_window: str = "7D"
    embargo_events: int = 0
    alphas: Sequence[float] = (0.1, 1.0, 10.0)
    # E3: Event-type aware purge windows (days)
    purge_window_by_event: Dict[str, int] = field(default_factory=dict)
    default_purge_days: int = 10
    # E2: Comprehensive trial counting dimensions
    n_feature_sets: int = 1
    n_models: int = 1
    n_hyperparams: int = 0  # auto-counted from alphas if 0
    n_label_windows: int = 1
    n_markets: int = 1

    def __post_init__(self) -> None:
        if not self.purge_window_by_event:
            from ..config import KALSHI_PURGE_WINDOW_BY_EVENT
            self.purge_window_by_event = dict(KALSHI_PURGE_WINDOW_BY_EVENT)
        if self.default_purge_days == 10:  # sentinel for "not explicitly set"
            from ..config import KALSHI_DEFAULT_PURGE_WINDOW
            self.default_purge_days = KALSHI_DEFAULT_PURGE_WINDOW


@dataclass
class EventWalkForwardFold:
    """Per-fold event walk-forward metrics for fit quality and event-return diagnostics."""
    fold_id: int
    train_event_count: int
    test_event_count: int
    best_alpha: float
    is_corr: float
    oos_corr: float
    oos_mse: float
    oos_mean_return: float
    oos_std_return: float
    worst_event_loss: float
    surprise_hit_rate: float


@dataclass
class EventWalkForwardResult:
    """Aggregate event walk-forward outputs and OOS traces used in promotion checks."""
    folds: List[EventWalkForwardFold]
    n_trials_total: int
    event_returns: List[float] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)
    release_timestamps: List[str] = field(default_factory=list)

    @property
    def wf_oos_corr(self) -> float:
        """wf oos corr."""
        if not self.folds:
            return np.nan
        vals = [f.oos_corr for f in self.folds if np.isfinite(f.oos_corr)]
        return float(np.mean(vals)) if vals else np.nan

    @property
    def wf_positive_fold_fraction(self) -> float:
        """wf positive fold fraction."""
        if not self.folds:
            return 0.0
        return float(np.mean([f.oos_mean_return > 0 for f in self.folds]))

    @property
    def wf_is_oos_gap(self) -> float:
        """wf is oos gap."""
        if not self.folds:
            return np.nan
        gaps = [max(0.0, f.is_corr - f.oos_corr) for f in self.folds if np.isfinite(f.is_corr) and np.isfinite(f.oos_corr)]
        return float(np.mean(gaps)) if gaps else np.nan

    @property
    def worst_event_loss(self) -> float:
        """worst event loss."""
        if not self.folds:
            return np.nan
        return float(np.min([f.worst_event_loss for f in self.folds]))

    def to_metrics(self) -> Dict[str, float]:
        """to metrics."""
        return {
            "wf_oos_corr": self.wf_oos_corr,
            "wf_positive_fold_fraction": self.wf_positive_fold_fraction,
            "wf_is_oos_gap": self.wf_is_oos_gap,
            "n_trials": float(self.n_trials_total),
            "worst_event_loss": self.worst_event_loss,
            "surprise_hit_rate": float(np.mean([f.surprise_hit_rate for f in self.folds])) if self.folds else np.nan,
        }


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 500,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate a bootstrap mean and 95% confidence interval for event returns."""
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(int(random_seed))
    idx = rng.randint(0, values.size, size=(int(max(n_bootstrap, 100)), values.size))
    samples = values[idx]
    means = np.nanmean(samples, axis=1)
    return float(np.nanmean(values)), float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def _event_regime_stability(
    event_returns: np.ndarray,
    event_types: Sequence[str],
) -> float:
    """Score return consistency across event types on a 0-1 stability scale."""
    if event_returns.size == 0 or not event_types:
        return np.nan
    n = min(event_returns.size, len(event_types))
    if n == 0:
        return np.nan
    df = pd.DataFrame(
        {
            "ret": event_returns[:n],
            "event_type": [str(x) for x in event_types[:n]],
        },
    )
    grouped = df.groupby("event_type")["ret"].mean()
    if len(grouped) == 0:
        return np.nan
    if len(grouped) == 1:
        return 1.0

    dispersion = float(np.nanstd(grouped.to_numpy(dtype=float)))
    scale = float(np.nanstd(df["ret"].to_numpy(dtype=float)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0
    return float(np.clip(np.exp(-dispersion / scale), 0.0, 1.0))


def evaluate_event_contract_metrics(
    result: EventWalkForwardResult,
    n_bootstrap: int = 500,
    max_events_per_day: float = 6.0,
) -> Dict[str, object]:
    """
    Advanced validation contract metrics for event strategies:
      - Deflated Sharpe (multiple-testing aware)
      - Monte Carlo robustness
      - Bootstrap mean-return stability
      - Turnover/capacity sanity proxies
      - Regime stability across event types
    """
    if result is None:
        return {}

    ret = np.asarray(result.event_returns, dtype=float)
    ret = ret[np.isfinite(ret)]
    if ret.size == 0:
        return {
            "dsr_significant": False,
            "dsr_p_value": 1.0,
            "dsr_deflated_sharpe": 0.0,
            "mc_p_value": 1.0,
            "mc_significant": False,
            "bootstrap_mean_return": np.nan,
            "bootstrap_mean_ci_low": np.nan,
            "bootstrap_mean_ci_high": np.nan,
            "capacity_constrained": True,
            "capacity_utilization": np.inf,
            "turnover_proxy": np.nan,
            "event_regime_stability": np.nan,
        }

    std = float(np.nanstd(ret))
    ann_factor = float(np.sqrt(252.0))
    observed_sharpe = float(np.nanmean(ret) / std * ann_factor) if std > 1e-12 else 0.0
    skew = float(pd.Series(ret).skew()) if ret.size > 3 else 0.0
    kurt = float(pd.Series(ret).kurt()) + 3.0 if ret.size > 3 else 3.0
    dsr = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=max(1, int(result.n_trials_total)),
        n_returns=int(ret.size),
        skewness=skew,
        kurtosis=kurt,
        annualization_factor=ann_factor,
    )

    mc = monte_carlo_validation(
        trade_returns=ret,
        n_simulations=int(max(n_bootstrap, 100)),
        holding_days=1,
        method="bootstrap",
    )

    mean_ret, ci_low, ci_high = _bootstrap_mean_ci(ret, n_bootstrap=n_bootstrap)

    pos = np.asarray(result.positions, dtype=float)
    pos = pos[np.isfinite(pos)]
    turnover_proxy = float(np.nanmean(np.abs(np.diff(pos)))) if pos.size > 1 else 0.0

    # T4: Deduplicate by timestamp to count unique events, not multi-horizon traces
    rel_ts = pd.to_datetime(pd.Series(result.release_timestamps), utc=True, errors="coerce").dropna()
    unique_rel_ts = rel_ts.drop_duplicates()
    if len(unique_rel_ts) > 1:
        span_days = max(1.0, float((unique_rel_ts.max() - unique_rel_ts.min()).total_seconds() / 86400.0))
        events_per_day = float(len(unique_rel_ts) / span_days)
    else:
        events_per_day = float(len(unique_rel_ts)) if len(unique_rel_ts) > 0 else 1.0

    cap_util = float(events_per_day / max(float(max_events_per_day), 1e-6))
    capacity_constrained = bool(cap_util > 1.0 or turnover_proxy > 1.5)

    regime_stability = _event_regime_stability(ret, result.event_types)

    return {
        "dsr_significant": bool(dsr.is_significant),
        "dsr_p_value": float(dsr.p_value),
        "dsr_deflated_sharpe": float(dsr.deflated_sharpe),
        "mc_p_value": float(mc.p_value),
        "mc_significant": bool(mc.is_significant),
        "bootstrap_mean_return": float(mean_ret),
        "bootstrap_mean_ci_low": float(ci_low),
        "bootstrap_mean_ci_high": float(ci_high),
        "capacity_constrained": capacity_constrained,
        "capacity_utilization": cap_util,
        "turnover_proxy": turnover_proxy,
        "event_regime_stability": float(regime_stability) if np.isfinite(regime_stability) else np.nan,
        "worst_event_loss": float(result.worst_event_loss),
        "surprise_hit_rate": float(np.mean([f.surprise_hit_rate for f in result.folds])) if result.folds else np.nan,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a finite-sample-safe correlation between two numeric arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 3:
        return np.nan
    aa = a[v]
    bb = b[v]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit a simple ridge regression coefficient vector for event walk-forward evaluation."""
    x = np.asarray(X, dtype=float)
    t = np.asarray(y, dtype=float)
    xtx = x.T @ x
    reg = float(alpha) * np.eye(xtx.shape[0])
    beta = np.linalg.pinv(xtx + reg) @ x.T @ t
    return beta


def _predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply a fitted linear coefficient vector to a feature matrix."""
    return np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float)


def _prepare_panel(
    panel: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
    label_col: str = "label_value",
) -> pd.DataFrame:
    """Normalize and merge feature panel and labels into a walk-forward-ready event dataset."""
    if panel is None or len(panel) == 0:
        return pd.DataFrame()

    p = panel.copy()
    if isinstance(p.index, pd.MultiIndex) and "event_id" in p.index.names:
        p = p.reset_index()

    if "event_id" not in p.columns:
        raise ValueError("panel must include event_id (column or index level)")
    if "release_ts" not in p.columns:
        raise ValueError("panel must include release_ts")

    if isinstance(labels, pd.Series):
        y = labels.rename(label_col).to_frame()
        y.index.name = "event_id"
        y = y.reset_index()
    else:
        y = labels.copy()
        if isinstance(y.index, pd.MultiIndex) and "event_id" in y.index.names:
            y = y.reset_index()
        elif y.index.name == "event_id":
            y = y.reset_index()

    if "event_id" not in y.columns:
        raise ValueError("labels must include event_id index or column")
    if label_col not in y.columns:
        numeric = [c for c in y.columns if c != "event_id" and pd.api.types.is_numeric_dtype(y[c])]
        if not numeric:
            raise ValueError(f"labels missing {label_col} and no numeric fallback column")
        y = y.rename(columns={numeric[0]: label_col})

    p = p.assign(
        event_id=p["event_id"].astype(str),
        release_ts=pd.to_datetime(p["release_ts"], utc=True, errors="coerce"),
    )
    y = y.assign(event_id=y["event_id"].astype(str))
    merged = p.merge(y[["event_id", label_col]], on="event_id", how="inner")
    merged = merged.assign(**{label_col: pd.to_numeric(merged[label_col], errors="coerce")})
    merged = merged[merged["release_ts"].notna() & merged[label_col].notna()].copy()
    return merged


def run_event_walkforward(
    panel: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
    config: Optional[EventWalkForwardConfig] = None,
    label_col: str = "label_value",
) -> EventWalkForwardResult:
    """Run purge/embargo-aware walk-forward evaluation on an event feature panel."""
    cfg = config or EventWalkForwardConfig()
    data = _prepare_panel(panel, labels, label_col=label_col)
    if data.empty:
        return EventWalkForwardResult(folds=[], n_trials_total=0)

    numeric_cols = [
        c for c in data.columns
        if c not in {"event_id", "release_ts", label_col, "event_type", "market_id", "horizon"}
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    if not numeric_cols:
        return EventWalkForwardResult(folds=[], n_trials_total=0)

    data = data.sort_values(["release_ts", "event_id"]).reset_index(drop=True)
    events = data[["event_id", "release_ts"]].drop_duplicates().sort_values("release_ts").reset_index(drop=True)

    purge_delta = pd.to_timedelta(cfg.purge_window)
    folds: List[EventWalkForwardFold] = []
    # E2: Comprehensive trial counting
    n_hyperparams = cfg.n_hyperparams if cfg.n_hyperparams > 0 else len(cfg.alphas)
    n_trials_total = 0
    n_trials_total_comprehensive = (
        max(1, cfg.n_feature_sets)
        * max(1, cfg.n_models)
        * max(1, n_hyperparams)
        * max(1, cfg.n_label_windows)
        * max(1, cfg.n_markets)
    )
    global_returns: List[float] = []
    global_positions: List[float] = []
    global_event_types: List[str] = []
    global_release_timestamps: List[str] = []

    start = int(cfg.train_min_events)
    while start + int(cfg.test_events_per_fold) <= len(events):
        test_slice = events.iloc[start : start + int(cfg.test_events_per_fold)]
        test_ids = set(test_slice["event_id"].astype(str).tolist())

        test_start = pd.Timestamp(test_slice["release_ts"].min())
        test_end = pd.Timestamp(test_slice["release_ts"].max())

        # E3: Event-type aware purge window
        if cfg.purge_window_by_event and "event_type" in data.columns:
            test_event_types_set = set(
                data[data["event_id"].isin(test_ids)]["event_type"].astype(str).str.upper().unique()
            )
            max_purge_days = max(
                (cfg.purge_window_by_event.get(et, cfg.default_purge_days) for et in test_event_types_set),
                default=cfg.default_purge_days,
            )
            effective_purge = pd.to_timedelta(f"{max_purge_days}D")
        else:
            effective_purge = purge_delta

        train_events = events.iloc[:start].copy()
        train_events = train_events[train_events["release_ts"] <= (test_start - effective_purge)]
        if int(cfg.embargo_events) > 0:
            train_events = train_events.iloc[:-int(cfg.embargo_events)] if len(train_events) > int(cfg.embargo_events) else train_events.iloc[0:0]

        train_ids = set(train_events["event_id"].astype(str).tolist())

        if len(train_ids) < max(10, int(cfg.train_min_events // 2)):
            start += int(cfg.step_events)
            continue

        train_df = data[data["event_id"].isin(train_ids)].copy()
        test_df = data[data["event_id"].isin(test_ids)].copy()
        if train_df.empty or test_df.empty:
            start += int(cfg.step_events)
            continue

        # Nested selection inside training window with purge gap (T2, SPEC_39).
        train_events_sorted = train_events.sort_values("release_ts")
        inner_cut = int(max(5, round(len(train_events_sorted) * 0.8)))

        # Apply purge gap between inner-train and inner-val
        inner_purge_events = max(1, int(cfg.embargo_events) or 1)
        inner_cut_purged = max(inner_cut - inner_purge_events, 5)

        inner_train_ids = set(
            train_events_sorted.iloc[:inner_cut_purged]["event_id"].astype(str).tolist()
        )
        inner_val_ids = set(
            train_events_sorted.iloc[inner_cut:]["event_id"].astype(str).tolist()
        )
        logger.debug(
            "Fold inner split: train=%d, purge_gap=%d, val=%d",
            len(inner_train_ids), inner_purge_events, len(inner_val_ids),
        )

        # If purge gap makes inner sets too small, use default alpha
        if len(inner_train_ids) < 5 or not inner_val_ids:
            best_alpha = float(cfg.alphas[len(cfg.alphas) // 2])
            logger.debug("Insufficient data for inner validation; using default alpha=%.2f", best_alpha)
            best_is = np.nan
        else:
            inner_train = train_df[train_df["event_id"].isin(inner_train_ids)]
            inner_val = train_df[train_df["event_id"].isin(inner_val_ids)]

            X_it = inner_train[numeric_cols].fillna(0.0).to_numpy(dtype=float)
            y_it = inner_train[label_col].to_numpy(dtype=float)
            X_iv = inner_val[numeric_cols].fillna(0.0).to_numpy(dtype=float)
            y_iv = inner_val[label_col].to_numpy(dtype=float)

            best_alpha = float(cfg.alphas[0]) if cfg.alphas else 1.0
            best_is = -np.inf
            for alpha in cfg.alphas:
                beta = _fit_ridge(X_it, y_it, alpha=float(alpha))
                pred_iv = _predict(X_iv, beta)
                score = _corr(pred_iv, y_iv)
                n_trials_total += 1
                if np.isfinite(score) and score > best_is:
                    best_is = float(score)
                    best_alpha = float(alpha)

        X_tr = train_df[numeric_cols].fillna(0.0).to_numpy(dtype=float)
        y_tr = train_df[label_col].to_numpy(dtype=float)
        X_te = test_df[numeric_cols].fillna(0.0).to_numpy(dtype=float)
        y_te = test_df[label_col].to_numpy(dtype=float)

        beta = _fit_ridge(X_tr, y_tr, alpha=best_alpha)
        pred = _predict(X_te, beta)

        oos_corr = _corr(pred, y_te)
        oos_mse = float(np.mean((pred - y_te) ** 2)) if len(y_te) > 0 else np.nan

        # Event strategy return proxy: sign(prediction) * realized outcome.
        pos_vec = np.sign(pred)
        strat_ret = pos_vec * y_te
        oos_mean_return = float(np.nanmean(strat_ret)) if len(strat_ret) > 0 else np.nan
        oos_std_return = float(np.nanstd(strat_ret)) if len(strat_ret) > 0 else np.nan
        worst_event_loss = float(np.nanmin(strat_ret)) if len(strat_ret) > 0 else np.nan

        # Surprise-conditional hit rate: top quartile |y| events.
        abs_y = np.abs(y_te)
        if len(abs_y) > 0:
            cutoff = float(np.nanquantile(abs_y, 0.75))
            mask = abs_y >= cutoff
            if mask.any():
                surprise_hit = float(np.mean(np.sign(pred[mask]) == np.sign(y_te[mask])))
            else:
                surprise_hit = np.nan
        else:
            surprise_hit = np.nan

        # Persist per-observation OOS traces for advanced event strategy validation.
        test_event_types = (
            test_df["event_type"].astype(str).tolist() if "event_type" in test_df.columns else [""] * len(strat_ret)
        )
        test_release_ts = (
            pd.to_datetime(test_df["release_ts"], utc=True, errors="coerce")
            .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            .fillna("")
            .tolist()
            if "release_ts" in test_df.columns
            else [""] * len(strat_ret)
        )
        global_returns.extend(np.asarray(strat_ret, dtype=float).tolist())
        global_positions.extend(np.asarray(pos_vec, dtype=float).tolist())
        global_event_types.extend(test_event_types[: len(strat_ret)])
        global_release_timestamps.extend(test_release_ts[: len(strat_ret)])

        folds.append(
            EventWalkForwardFold(
                fold_id=len(folds) + 1,
                train_event_count=int(len(train_ids)),
                test_event_count=int(len(test_ids)),
                best_alpha=best_alpha,
                is_corr=float(best_is) if np.isfinite(best_is) else np.nan,
                oos_corr=float(oos_corr) if np.isfinite(oos_corr) else np.nan,
                oos_mse=float(oos_mse) if np.isfinite(oos_mse) else np.nan,
                oos_mean_return=float(oos_mean_return) if np.isfinite(oos_mean_return) else np.nan,
                oos_std_return=float(oos_std_return) if np.isfinite(oos_std_return) else np.nan,
                worst_event_loss=float(worst_event_loss) if np.isfinite(worst_event_loss) else np.nan,
                surprise_hit_rate=float(surprise_hit) if np.isfinite(surprise_hit) else np.nan,
            ),
        )

        start += int(cfg.step_events)

    return EventWalkForwardResult(
        folds=folds,
        n_trials_total=max(n_trials_total, n_trials_total_comprehensive),
        event_returns=global_returns,
        positions=global_positions,
        event_types=global_event_types,
        release_timestamps=global_release_timestamps,
    )
