"""
Evaluation Visualization — chart generation for HTML reports.

Uses plotly for interactive charts with a matplotlib fallback.
All functions return a dict with 'html' (plotly div) and 'data' (raw
values for JSON export).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def plot_regime_slices(
    slice_metrics: Dict[str, Dict],
) -> Dict:
    """Bar chart comparing Sharpe ratio across regime slices.

    Parameters
    ----------
    slice_metrics : dict[str, dict]
        Mapping from slice name to metrics dict (must have 'sharpe' key).

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    names = list(slice_metrics.keys())
    sharpes = [m.get("sharpe", 0.0) for m in slice_metrics.values()]
    n_samples = [m.get("n_samples", 0) for m in slice_metrics.values()]

    data = {"names": names, "sharpes": sharpes, "n_samples": n_samples}

    if not _HAS_PLOTLY or not names:
        return {"html": None, "data": data}

    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=sharpes,
        marker_color=colors,
        text=[f"N={n}" for n in n_samples],
        textposition="outside",
    ))
    fig.update_layout(
        title="Sharpe Ratio by Regime Slice",
        yaxis_title="Annualized Sharpe",
        xaxis_title="Slice",
        template="plotly_white",
        height=400,
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}


def plot_rolling_ic(
    ic_series: pd.Series,
    decay_threshold: float = 0.02,
) -> Dict:
    """Line chart of rolling information coefficient with decay threshold.

    Parameters
    ----------
    ic_series : pd.Series
        Rolling IC values indexed by date.
    decay_threshold : float
        Horizontal threshold line.

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    ic_clean = ic_series.dropna()
    data = {
        "dates": [str(d) for d in ic_clean.index],
        "ic_values": ic_clean.values.tolist(),
        "decay_threshold": decay_threshold,
    }

    if not _HAS_PLOTLY or len(ic_clean) == 0:
        return {"html": None, "data": data}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ic_clean.index,
        y=ic_clean.values,
        mode="lines",
        name="Rolling IC",
        line=dict(color="#3498db", width=1.5),
    ))
    fig.add_hline(
        y=decay_threshold, line_dash="dash",
        annotation_text=f"Threshold ({decay_threshold})",
        line_color="#e74c3c",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Rolling Information Coefficient",
        yaxis_title="Spearman IC",
        xaxis_title="Date",
        template="plotly_white",
        height=350,
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}


def plot_decile_spread(
    decile_returns: List[float],
    decile_counts: Optional[List[int]] = None,
) -> Dict:
    """Bar chart of mean return per prediction decile.

    Parameters
    ----------
    decile_returns : list[float]
        Mean return per decile (index 0 = lowest prediction).
    decile_counts : list[int], optional
        Number of observations per decile.

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    n_q = len(decile_returns)
    labels = [f"D{i + 1}" for i in range(n_q)]
    data = {
        "labels": labels,
        "returns": decile_returns,
        "counts": decile_counts or [0] * n_q,
    }

    if not _HAS_PLOTLY or n_q == 0:
        return {"html": None, "data": data}

    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in decile_returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=decile_returns,
        marker_color=colors,
        text=[f"{r:.4f}" for r in decile_returns],
        textposition="outside",
    ))
    fig.update_layout(
        title="Mean Return by Prediction Decile",
        yaxis_title="Mean Return",
        xaxis_title="Decile (D1=lowest prediction, D10=highest)",
        template="plotly_white",
        height=400,
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}


def plot_underwater(
    returns: pd.Series,
) -> Dict:
    """Underwater (drawdown) plot from a return series.

    Parameters
    ----------
    returns : pd.Series
        Per-bar return series.

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    ret_arr = returns.values.astype(float)
    cum_eq = np.cumprod(1 + ret_arr)
    running_max = np.maximum.accumulate(cum_eq)
    drawdowns = (cum_eq - running_max) / np.where(running_max > 0, running_max, 1.0)

    data = {
        "dates": [str(d) for d in returns.index],
        "drawdowns": drawdowns.tolist(),
    }

    if not _HAS_PLOTLY or len(returns) == 0:
        return {"html": None, "data": data}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns.index,
        y=drawdowns,
        fill="tozeroy",
        mode="lines",
        line=dict(color="#e74c3c", width=1),
        fillcolor="rgba(231, 76, 60, 0.3)",
    ))
    fig.update_layout(
        title="Underwater Plot (Drawdown from Peak)",
        yaxis_title="Drawdown",
        xaxis_title="Date",
        template="plotly_white",
        height=350,
        yaxis=dict(tickformat=".0%"),
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}


def plot_recovery_distribution(
    recovery_times: pd.Series,
) -> Dict:
    """Histogram of recovery times.

    Parameters
    ----------
    recovery_times : pd.Series
        Recovery times per drawdown episode.

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    rt_vals = recovery_times.dropna().values.astype(float)
    data = {"recovery_times": rt_vals.tolist()}

    if not _HAS_PLOTLY or len(rt_vals) == 0:
        return {"html": None, "data": data}

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rt_vals,
        nbinsx=min(30, max(5, len(rt_vals) // 3)),
        marker_color="#3498db",
    ))
    fig.update_layout(
        title="Recovery Time Distribution",
        yaxis_title="Count",
        xaxis_title="Recovery Time (bars)",
        template="plotly_white",
        height=350,
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}


def plot_calibration_curve(
    reliability_curve: Dict,
) -> Dict:
    """Calibration (reliability) curve: predicted vs observed frequency.

    Parameters
    ----------
    reliability_curve : dict
        From ``compute_reliability_curve`` — keys: 'bin_centers',
        'observed_freq', 'avg_predicted', 'bin_counts'.

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    bin_centers = reliability_curve.get("bin_centers", [])
    obs_freq = reliability_curve.get("observed_freq", [])
    avg_pred = reliability_curve.get("avg_predicted", [])
    counts = reliability_curve.get("bin_counts", [])

    data = {
        "bin_centers": bin_centers,
        "observed_freq": obs_freq,
        "avg_predicted": avg_pred,
        "bin_counts": counts,
    }

    if not _HAS_PLOTLY or not bin_centers:
        return {"html": None, "data": data}

    # Filter out NaN bins
    valid_idx = [
        i for i in range(len(bin_centers))
        if not (np.isnan(obs_freq[i]) if obs_freq[i] is not None else True)
    ]

    if not valid_idx:
        return {"html": None, "data": data}

    fig = go.Figure()

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect calibration",
    ))

    # Actual calibration curve
    fig.add_trace(go.Scatter(
        x=[avg_pred[i] for i in valid_idx],
        y=[obs_freq[i] for i in valid_idx],
        mode="lines+markers",
        name="Model calibration",
        line=dict(color="#3498db", width=2),
        marker=dict(size=8),
        text=[f"N={counts[i]}" for i in valid_idx],
    ))

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Frequency",
        template="plotly_white",
        height=400,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}


def plot_walk_forward_folds(
    folds: List[Dict],
) -> Dict:
    """Scatter plot of IS vs OOS Sharpe per walk-forward fold.

    Parameters
    ----------
    folds : list[dict]
        Each dict must have 'train_sharpe' and 'test_sharpe' keys.

    Returns
    -------
    dict
        Keys: 'html' (str or None), 'data' (dict).
    """
    is_sharpes = [f.get("train_sharpe", f.get("train_sharpe", 0.0)) for f in folds]
    oos_sharpes = [f.get("test_sharpe", f.get("test_sharpe", 0.0)) for f in folds]

    data = {"is_sharpes": is_sharpes, "oos_sharpes": oos_sharpes}

    if not _HAS_PLOTLY or not folds:
        return {"html": None, "data": data}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=is_sharpes,
        y=oos_sharpes,
        mode="markers",
        marker=dict(size=8, color="#3498db"),
        name="Folds",
    ))

    # 45-degree line (no overfitting)
    all_vals = is_sharpes + oos_sharpes
    lo, hi = min(all_vals) - 0.1, max(all_vals) + 0.1
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="No overfit",
    ))

    fig.update_layout(
        title="Walk-Forward: In-Sample vs Out-of-Sample Sharpe",
        xaxis_title="In-Sample Sharpe",
        yaxis_title="Out-of-Sample Sharpe",
        template="plotly_white",
        height=400,
    )

    return {"html": fig.to_html(full_html=False, include_plotlyjs="cdn"), "data": data}
