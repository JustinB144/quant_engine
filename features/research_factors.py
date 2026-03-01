"""
Research-derived factor construction for quant_engine.

This module adds production-oriented features inspired by the papers in
`QUANT RESEARCH`:
1) Order-flow imbalance and depth-adjusted impact proxies.
2) Markov state features for queue-imbalance style dynamics.
3) Vol-scaled time-series momentum.
4) Signature-inspired path features (level 1 / level 2 + Levy area).
5) Volatility term-structure factors (KL/PCA style decomposition).
6) Cross-asset lead-lag network momentum and volatility spillover metrics.
7) DTW-based lead-lag detection for cross-asset analysis.
8) Path signature features (truncated order-2) for (price, volume) paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True)
class ResearchFactorConfig:
    """Configuration for research-derived factor generation."""

    ofi_window: int = 20
    queue_window: int = 63
    queue_state_threshold: float = 0.25
    tsmom_lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    tsmom_vol_window: int = 60
    signature_window: int = 20
    vol_surface_pca_window: int = 252
    network_window: int = 63
    network_min_obs: int = 40
    cross_asset_lag_bars: int = 1


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Causal rolling z-score."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + _EPS)


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Internal helper for safe pct change."""
    out = series.pct_change(periods)
    return out.replace([np.inf, -np.inf], np.nan)


def _required_ohlcv(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Internal helper for required ohlcv."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    open_ = pd.to_numeric(df["Open"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    return close, open_, high, low, volume


def compute_order_flow_impact_factors(
    df: pd.DataFrame,
    config: ResearchFactorConfig,
) -> pd.DataFrame:
    """
    Order-flow imbalance and price-impact proxies (Cont et al. inspired).

    Uses only OHLCV (no LOB feed), so these are practical proxies:
    - signed volume imbalance (improved: candle body ratio instead of sign)
    - depth-adjusted imbalance
    - rolling impact slope lambda
    - square-root impact transform
    - normalized OFI, OFI persistence (autocorrelation), OFI momentum
    """
    close, open_, high, low, volume = _required_ohlcv(df)
    w = config.ofi_window
    minp = max(5, w // 3)

    ret_1d = _safe_pct_change(close, 1)

    # Improved OFI: use candle body ratio instead of sign(close_change)
    # (close - open) / max(high - low, 1e-10) captures direction AND conviction
    candle_range = (high - low).clip(lower=1e-10)
    signed_direction = ((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0)

    ofi_raw = signed_direction * volume
    vol_avg = volume.rolling(w, min_periods=minp).mean()
    ofi_norm = ofi_raw / (vol_avg + _EPS)

    dollar_volume = close * volume
    depth = dollar_volume.rolling(w, min_periods=minp).mean()
    ofi_depth_adj = ofi_raw / np.sqrt(depth + _EPS)

    mean_r = ret_1d.rolling(w, min_periods=minp).mean()
    mean_ofi = ofi_norm.rolling(w, min_periods=minp).mean()
    cov = (ret_1d * ofi_norm).rolling(w, min_periods=minp).mean() - (mean_r * mean_ofi)
    var_ofi = ofi_norm.rolling(w, min_periods=minp).var()
    impact_lambda = cov / (var_ofi + _EPS)

    impact_linear = impact_lambda * ofi_norm
    impact_sqrt = impact_lambda * np.sign(ofi_norm) * np.sqrt(np.abs(ofi_norm))

    # Additional OFI features
    # ofi_normalized: OFI divided by rolling average volume (already ofi_norm)
    ofi_normalized = ofi_norm

    # ofi_persistence: autocorrelation of OFI over 5 bars
    ofi_persistence = ofi_norm.rolling(5, min_periods=3).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 3 else np.nan,
        raw=False,
    )

    # ofi_momentum: cumulative OFI over 5 bars
    ofi_momentum = ofi_norm.rolling(5, min_periods=3).sum()

    out = pd.DataFrame(index=df.index)
    out[f"OFI_{w}"] = ofi_norm
    out[f"OFI_Z_{w}"] = _rolling_zscore(ofi_norm, window=w, min_periods=minp)
    out[f"OFI_DepthAdj_{w}"] = ofi_depth_adj
    out[f"ImpactLambda_{w}"] = impact_lambda
    out[f"ImpactLinear_{w}"] = impact_linear
    out[f"ImpactSqrt_{w}"] = impact_sqrt
    out["ofi_normalized"] = ofi_normalized
    out["ofi_persistence"] = ofi_persistence
    out["ofi_momentum"] = ofi_momentum
    return out


def compute_markov_queue_features(
    df: pd.DataFrame,
    config: ResearchFactorConfig,
) -> pd.DataFrame:
    """
    Markov-style queue imbalance features (de Larrard style state framing).

    Queue imbalance proxy is inferred from candle geometry:
        (Close - Open) / (High - Low)
    """
    close, open_, high, low, _ = _required_ohlcv(df)
    w = config.queue_window
    minp = max(10, w // 3)
    threshold = float(config.queue_state_threshold)

    q_proxy = ((close - open_) / (high - low + _EPS)).clip(-1.0, 1.0).fillna(0.0)
    state = pd.Series(
        np.where(q_proxy > threshold, 1, np.where(q_proxy < -threshold, -1, 0)),
        index=df.index,
        dtype=int,
    )

    ret_1d = _safe_pct_change(close, 1)
    up_move = ret_1d > 0
    down_move = ret_1d < 0

    prev_state = state.shift(1)

    def _cond_prob(prev_code: int, move_mask: pd.Series) -> pd.Series:
        base = (prev_state == prev_code).astype(float)
        num = (base * move_mask.astype(float)).rolling(w, min_periods=minp).sum()
        den = base.rolling(w, min_periods=minp).sum()
        return num / (den + _EPS)

    p_up_neg = _cond_prob(-1, up_move)
    p_up_neu = _cond_prob(0, up_move)
    p_up_pos = _cond_prob(1, up_move)
    p_dn_neg = _cond_prob(-1, down_move)
    p_dn_neu = _cond_prob(0, down_move)
    p_dn_pos = _cond_prob(1, down_move)

    p_up_state = pd.Series(np.nan, index=df.index, dtype=float)
    p_dn_state = pd.Series(np.nan, index=df.index, dtype=float)
    p_up_state[state == -1] = p_up_neg[state == -1]
    p_up_state[state == 0] = p_up_neu[state == 0]
    p_up_state[state == 1] = p_up_pos[state == 1]
    p_dn_state[state == -1] = p_dn_neg[state == -1]
    p_dn_state[state == 0] = p_dn_neu[state == 0]
    p_dn_state[state == 1] = p_dn_pos[state == 1]

    p_flat = (1.0 - p_up_state.fillna(0.0) - p_dn_state.fillna(0.0)).clip(0.0, 1.0)
    transition_entropy = -(
        p_up_state.fillna(0.0) * np.log(p_up_state.fillna(0.0) + _EPS)
        + p_dn_state.fillna(0.0) * np.log(p_dn_state.fillna(0.0) + _EPS)
        + p_flat * np.log(p_flat + _EPS)
    )

    group_id = state.ne(state.shift(1)).cumsum()
    state_duration = group_id.groupby(group_id).cumcount() + 1

    out = pd.DataFrame(index=df.index)
    out["QueueImbalance_Proxy"] = q_proxy
    out["LOB_State"] = state.astype(float)
    out[f"LOB_PUp_State_{w}"] = p_up_state
    out[f"LOB_PDown_State_{w}"] = p_dn_state
    out[f"LOB_TransitionEntropy_{w}"] = transition_entropy
    out["LOB_StateDuration"] = state_duration.astype(float)
    return out


def compute_time_series_momentum_factors(
    df: pd.DataFrame,
    config: ResearchFactorConfig,
) -> pd.DataFrame:
    """
    Vol-scaled time-series momentum factors (Moskowitz/Ooi/Pedersen style).

    All momentum computations are **backward-looking** (causal):
        mom_h = close / close.shift(h) - 1
    This compares the current price to the price h bars ago.

    Column naming uses ``TSMom_lag{h}`` to explicitly signal that these
    are lagged (causal) features, not forward-looking.
    """
    close, _, _, _, _ = _required_ohlcv(df)
    lookbacks = tuple(int(x) for x in config.tsmom_lookbacks if int(x) > 1)
    vol_window = config.tsmom_vol_window

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(vol_window, min_periods=max(20, vol_window // 3)).std() * np.sqrt(252.0)

    out = pd.DataFrame(index=df.index)
    lb_cols: list[str] = []
    for lb in lookbacks:
        # Backward-looking momentum: close / close.shift(lb) - 1
        raw = _safe_pct_change(close, lb)
        scaled = raw / (vol * np.sqrt(lb / 252.0) + _EPS)
        col = f"TSMom_lag{lb}"
        out[col] = scaled
        lb_cols.append(col)

    # 12-month minus 1-month momentum (already backward-looking)
    # shift(21) = 21 bars ago, shift(252) = 252 bars ago
    out["TSMom_12m1m"] = close.shift(21) / close.shift(252) - 1.0

    if lb_cols:
        mat = out[lb_cols]
        out["TSMom_Ensemble"] = mat.mean(axis=1)
        pos = (mat > 0).sum(axis=1)
        neg = (mat < 0).sum(axis=1)
        out["TSMom_SignAgreement"] = (pos - neg) / max(len(lb_cols), 1)
    else:
        out["TSMom_Ensemble"] = np.nan
        out["TSMom_SignAgreement"] = np.nan

    return out


def compute_vol_scaled_momentum(
    df: pd.DataFrame,
    horizons: List[int] = [5, 10, 20, 60, 120, 252],
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Volatility-scaled time-series momentum enhancements.

    Inspired by ssrn-2089463 (Moskowitz et al.).  This extends classic TSMOM
    with regime-aware normalization, reversal-horizon detection, and
    cross-sectional (relative) momentum.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with at least a 'Close' column.
    horizons : list of int
        Look-back horizons (in bars) for momentum computation.
    vol_window : int
        Trailing window for realized volatility used in scaling.

    Returns
    -------
    pd.DataFrame
        Columns: vsmom_{h} for each horizon, reversal_horizon,
        momentum_persistence, relative_mom_10, relative_mom_20,
        relative_mom_60.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    returns = close.pct_change()
    vol = returns.rolling(vol_window, min_periods=max(5, vol_window // 3)).std()

    out = pd.DataFrame(index=df.index)

    # --- Raw and vol-scaled momentum at each horizon ---
    mom_dict: dict[int, pd.Series] = {}
    for h in horizons:
        mom_h = close / close.shift(h) - 1
        mom_dict[h] = mom_h
        # Normalize: vsmom_h = mom_h / (vol * sqrt(h))
        vsmom_h = mom_h / (vol * np.sqrt(h) + _EPS)
        out[f"vsmom_{h}"] = vsmom_h

    # --- Reversal horizon detection ---
    # For each row, find the first horizon where momentum sign flips
    # (changes from positive to negative compared to the shortest horizon).
    # If momentum never flips, reversal_horizon = max(horizons) + 1.
    sorted_horizons = sorted(horizons)
    mom_signs = pd.DataFrame(
        {h: np.sign(mom_dict[h]) for h in sorted_horizons},
        index=df.index,
    )
    # Reference sign = sign at the shortest horizon
    ref_sign = mom_signs[sorted_horizons[0]]

    reversal_horizon = pd.Series(
        float(sorted_horizons[-1] + 1), index=df.index, dtype=float
    )
    for h in sorted_horizons[1:]:
        # Where sign differs from the reference and we haven't found a
        # reversal yet (reversal_horizon still at the sentinel value)
        sign_flip = (mom_signs[h] != ref_sign) & (mom_signs[h] != 0) & (ref_sign != 0)
        still_sentinel = reversal_horizon == float(sorted_horizons[-1] + 1)
        reversal_horizon = reversal_horizon.where(
            ~(sign_flip & still_sentinel), float(h)
        )
    out["reversal_horizon"] = reversal_horizon

    # --- Momentum persistence ---
    # Fraction of horizons where raw momentum is positive
    positive_count = pd.DataFrame(
        {h: (mom_dict[h] > 0).astype(float) for h in sorted_horizons},
        index=df.index,
    )
    out["momentum_persistence"] = positive_count.mean(axis=1)

    # --- Cross-sectional (relative) momentum ---
    # In a single-asset context the "market" momentum is just the stock's own,
    # so relative_mom is zero.  However this function is designed to be called
    # within a universe loop; the caller can subtract the cross-sectional mean
    # afterwards.  Here we store raw momentum for the three key horizons so
    # the pipeline can compute the cross-sectional adjustment.
    for h in [10, 20, 60]:
        if h in mom_dict:
            out[f"relative_mom_{h}"] = mom_dict[h]
        else:
            # Compute it fresh if the horizon was not in the main list
            out[f"relative_mom_{h}"] = close / close.shift(h) - 1

    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_levy_area(dx: np.ndarray, dy: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Rolling Levy area for a 2D path of increments.

    Complexity is O(n * window) with small windows (default 20), which is
    practical for the intended use.
    """
    n = len(dx)
    out = np.full(n, np.nan, dtype=float)
    for end in range(window - 1, n):
        start = end - window + 1
        x = dx[start : end + 1]
        y = dy[start : end + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < min_periods:
            continue
        xv = x[valid]
        yv = y[valid]
        if len(xv) < 2:
            continue
        cx = np.cumsum(xv)
        cy = np.cumsum(yv)
        out[end] = 0.5 * np.sum(cx[:-1] * yv[1:] - cy[:-1] * xv[1:])
    return out


def compute_signature_path_features(
    df: pd.DataFrame,
    config: ResearchFactorConfig,
) -> pd.DataFrame:
    """
    Signature-inspired path features for returns-volume trajectory.

    Uses truncated level-1 and level-2 style quantities plus Levy area.
    """
    close, _, _, _, volume = _required_ohlcv(df)
    w = config.signature_window
    minp = max(6, w // 3)

    dx = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    dy = np.log1p(volume).diff().replace([np.inf, -np.inf], np.nan)

    s1_x = dx.rolling(w, min_periods=minp).sum()
    s1_y = dy.rolling(w, min_periods=minp).sum()
    levy = _rolling_levy_area(dx.values.astype(float), dy.values.astype(float), w, min_periods=minp)
    levy_s = pd.Series(levy, index=df.index)

    s2_xx = 0.5 * (s1_x**2)
    s2_yy = 0.5 * (s1_y**2)
    s2_xy = 0.5 * (s1_x * s1_y + levy_s)

    out = pd.DataFrame(index=df.index)
    out[f"SigL1_Return_{w}"] = s1_x
    out[f"SigL1_Volume_{w}"] = s1_y
    out[f"SigL2_XX_{w}"] = s2_xx
    out[f"SigL2_XY_{w}"] = s2_xy
    out[f"SigL2_YY_{w}"] = s2_yy
    out[f"SigLevyArea_{w}"] = levy_s
    return out


def compute_vol_surface_factors(
    df: pd.DataFrame,
    config: ResearchFactorConfig,
) -> pd.DataFrame:
    """
    Volatility term-structure factors inspired by implied-vol surface dynamics.

    Without options data, this uses realized volatility across maturities
    and a rolling PCA/KL decomposition of term-structure changes.
    """
    close, _, _, _, _ = _required_ohlcv(df)
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

    term_windows = (5, 10, 21, 63)
    vol_curve = pd.DataFrame(
        {
            f"RV_{w}": log_ret.rolling(w, min_periods=max(3, w // 2)).std() * np.sqrt(252.0)
            for w in term_windows
        },
        index=df.index,
    )

    out = pd.DataFrame(index=df.index)
    out["VolSurf_Level"] = vol_curve.mean(axis=1)
    out["VolSurf_Slope"] = vol_curve["RV_5"] - vol_curve["RV_63"]
    out["VolSurf_Curvature"] = vol_curve["RV_10"] - 2.0 * vol_curve["RV_21"] + vol_curve["RV_63"]

    changes = vol_curve.diff()
    pca_window = config.vol_surface_pca_window
    min_obs = max(40, pca_window // 3)

    pc1 = np.full(len(changes), np.nan, dtype=float)
    pc2 = np.full(len(changes), np.nan, dtype=float)
    pc3 = np.full(len(changes), np.nan, dtype=float)

    vals = changes.values.astype(float)
    for i in range(pca_window, len(changes)):
        block = vals[i - pca_window + 1 : i + 1]
        valid_rows = np.all(np.isfinite(block), axis=1)
        if valid_rows.sum() < min_obs:
            continue
        mat = block[valid_rows]
        if mat.shape[0] < mat.shape[1] + 2:
            continue

        cov = np.cov(mat, rowvar=False)
        if not np.all(np.isfinite(cov)):
            continue

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        current = vals[i]
        if not np.all(np.isfinite(current)):
            continue
        scores = current @ eigvecs
        pc1[i] = scores[0]
        if len(scores) > 1:
            pc2[i] = scores[1]
        if len(scores) > 2:
            pc3[i] = scores[2]

    out["VolSurf_PC1"] = pc1
    out["VolSurf_PC2"] = pc2
    out["VolSurf_PC3"] = pc3
    return out


def compute_single_asset_research_factors(
    df: pd.DataFrame,
    config: ResearchFactorConfig | None = None,
) -> pd.DataFrame:
    """Compute all single-asset research factors."""
    cfg = config or ResearchFactorConfig()
    parts = [
        compute_order_flow_impact_factors(df, cfg),
        compute_markov_queue_features(df, cfg),
        compute_time_series_momentum_factors(df, cfg),
        compute_vol_scaled_momentum(df),
        compute_signature_path_features(df, cfg),
        compute_vol_surface_factors(df, cfg),
    ]
    out = pd.concat(parts, axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def _standardize_block(block: np.ndarray) -> np.ndarray:
    """Column-wise z-score with NaN-safe handling."""
    mu = np.nanmean(block, axis=0, keepdims=True)
    sd = np.nanstd(block, axis=0, ddof=1, keepdims=True)
    z = np.divide(block - mu, sd + _EPS, out=np.zeros_like(block), where=np.isfinite(sd))
    z[~np.isfinite(z)] = 0.0
    return z


def _lagged_weight_matrix(
    values: np.ndarray,
    t: int,
    window: int,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build positive lagged correlation weights:
        follower_t ~ leader_{t-1}
    """
    n_assets = values.shape[1]
    x = values[t - window + 1 : t + 1]  # follower window
    y = values[t - window : t]  # lagged leader window

    finite_x = np.isfinite(x).sum(axis=0)
    finite_y = np.isfinite(y).sum(axis=0)
    valid_cols = (finite_x >= min_obs) & (finite_y >= min_obs)
    valid_idx = np.flatnonzero(valid_cols)

    if len(valid_idx) < 2:
        return np.zeros((n_assets, n_assets), dtype=float), valid_cols

    x_sub = x[:, valid_idx]
    y_sub = y[:, valid_idx]
    xz = _standardize_block(x_sub)
    yz = _standardize_block(y_sub)
    corr = (xz.T @ yz) / max(xz.shape[0] - 1, 1)
    np.fill_diagonal(corr, 0.0)

    # Keep directional lead-lag influence weights and row-normalize.
    weights = np.clip(corr, 0.0, None)
    row_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, row_sum + _EPS, out=np.zeros_like(weights), where=row_sum > 0.0)

    full = np.zeros((n_assets, n_assets), dtype=float)
    full[np.ix_(valid_idx, valid_idx)] = weights
    return full, valid_cols


def compute_cross_asset_research_factors(
    price_data: Mapping[str, pd.DataFrame],
    config: ResearchFactorConfig | None = None,
) -> pd.DataFrame:
    """
    Compute cross-asset network momentum and volatility spillover factors.

    Returns:
        MultiIndex DataFrame indexed by (permno, date).
    """
    cfg = config or ResearchFactorConfig()
    close_dict: dict[str, pd.Series] = {}
    for permno, df in price_data.items():
        if "Close" not in df.columns or len(df) == 0:
            continue
        s = pd.to_numeric(df["Close"], errors="coerce")
        s.index = pd.to_datetime(s.index)
        close_dict[str(permno)] = s

    if len(close_dict) < 2:
        return pd.DataFrame()

    close = pd.DataFrame(close_dict).sort_index().dropna(axis=1, how="all")
    if close.shape[1] < 2 or close.shape[0] < cfg.network_window + 2:
        return pd.DataFrame()

    permnos = list(close.columns)
    dates = close.index
    n_dates, n_assets = close.shape

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    rv = ret.rolling(5, min_periods=3).std() * np.sqrt(252.0)
    rv_innov = rv - rv.ewm(span=30, min_periods=10).mean()

    net_mom = np.full((n_dates, n_assets), np.nan, dtype=float)
    lead_cent = np.full((n_dates, n_assets), np.nan, dtype=float)
    recv_cent = np.full((n_dates, n_assets), np.nan, dtype=float)
    graph_density = np.full((n_dates, n_assets), np.nan, dtype=float)

    vol_in = np.full((n_dates, n_assets), np.nan, dtype=float)
    vol_out = np.full((n_dates, n_assets), np.nan, dtype=float)

    ret_vals = ret.values.astype(float)
    rv_vals = rv_innov.values.astype(float)
    window = cfg.network_window
    min_obs = min(cfg.network_min_obs, max(6, window - 1))

    for t in range(window, n_dates):
        w_ret, valid_ret = _lagged_weight_matrix(ret_vals, t, window=window, min_obs=min_obs)
        if valid_ret.sum() >= 2:
            leaders = np.nan_to_num(ret_vals[t - 1], nan=0.0)
            net = w_ret @ leaders
            net[~valid_ret] = np.nan
            net_mom[t] = net

            recv = w_ret.sum(axis=1)
            lead = w_ret.sum(axis=0)
            recv[~valid_ret] = np.nan
            lead[~valid_ret] = np.nan
            recv_cent[t] = recv
            lead_cent[t] = lead

            edge_count = float((w_ret > 0).sum())
            density = edge_count / max(float(n_assets * max(n_assets - 1, 1)), 1.0)
            graph_density[t, valid_ret] = density

        w_vol, valid_vol = _lagged_weight_matrix(rv_vals, t, window=window, min_obs=min_obs)
        if valid_vol.sum() >= 2:
            vol_leaders = np.nan_to_num(rv_vals[t - 1], nan=0.0)
            vin = w_vol @ np.abs(vol_leaders)
            vout = w_vol.sum(axis=0)
            vin[~valid_vol] = np.nan
            vout[~valid_vol] = np.nan
            vol_in[t] = vin
            vol_out[t] = vout

    # Keep NaN where ticker data itself is unavailable on that date.
    valid_mask = close.notna().values
    for arr in (net_mom, lead_cent, recv_cent, graph_density, vol_in, vol_out):
        arr[~valid_mask] = np.nan

    lag = max(0, int(cfg.cross_asset_lag_bars))
    panel = []
    for i, permno in enumerate(permnos):
        f = pd.DataFrame(
            {
                "NetMom_Spillover": net_mom[:, i],
                "NetMom_LeadCentrality": lead_cent[:, i],
                "NetMom_RecvCentrality": recv_cent[:, i],
                "NetMom_GraphDensity": graph_density[:, i],
                "VolSpillover_In": vol_in[:, i],
                "VolSpillover_Out": vol_out[:, i],
                "VolSpillover_Net": vol_out[:, i] - vol_in[:, i],
            },
            index=dates,
        )
        f.index.name = "date"
        if lag > 0:
            # Enforce causal usage of cross-asset context by lagging the full block.
            f = f.shift(lag)
        f["permno"] = permno
        f = f.set_index("permno", append=True).reorder_levels([1, 0])
        panel.append(f)

    out = pd.concat(panel).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 7) DTW-based lead-lag detection
# ---------------------------------------------------------------------------

def _dtw_distance_numpy(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Pure numpy DTW distance computation using dynamic programming.

    Returns (distance, path) where path is a list of (i, j) index pairs
    representing the optimal alignment.
    """
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    D[0, 0] = 0.0

    # Cost matrix (squared Euclidean)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack to find alignment path
    path: list = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path.append((i, j))
        elif j == 0:
            i -= 1
            path.append((i, j))
        else:
            candidates = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
            argmin = int(np.argmin(candidates))
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
            path.append((i, j))
    assert all(pi >= 0 and pj >= 0 for pi, pj in path), "Negative indices in DTW path"
    path.reverse()
    return float(np.sqrt(D[n, m])), path


def _dtw_avg_lag_from_path(path: list) -> float:
    """Extract average lag from DTW alignment path.

    Positive value means the first series (x) leads the second (y).
    """
    if not path:
        return 0.0
    lags = [float(i - j) for i, j in path]
    return float(np.mean(lags))


def compute_dtw_lead_lag(
    returns: Dict[str, pd.Series],
    window: int = 60,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    DTW-based lead-lag detection across a universe of assets.

    For each pair of assets, computes DTW distance and alignment path within
    rolling windows.  From the alignment path, extracts the average lag
    (which asset leads).

    Parameters
    ----------
    returns : dict of {asset_name: pd.Series}
        Return series for each asset, all sharing a common DatetimeIndex.
    window : int
        Rolling window size for DTW computation.
    max_lag : int
        Maximum lag (in bars) to consider for DTW sub-sequences.  The DTW
        is computed on windows of length ``max_lag`` within each rolling
        block for efficiency.

    Returns
    -------
    pd.DataFrame
        MultiIndex (permno, date) with columns:
        ``dtw_leader_score``, ``dtw_follower_score``, ``dtw_avg_lag``.
    """
    # Try fast external libraries first, fall back to pure numpy
    _dtw_func = _dtw_distance_numpy
    try:
        from dtaidistance import dtw as dtai_dtw  # type: ignore

        def _dtw_func_dtai(x: np.ndarray, y: np.ndarray) -> tuple:
            dist = dtai_dtw.distance(x.astype(np.double), y.astype(np.double))
            path = dtai_dtw.warping_path(x.astype(np.double), y.astype(np.double))
            return float(dist), path

        _dtw_func = _dtw_func_dtai
    except ImportError:
        try:
            from tslearn.metrics import dtw_path as tslearn_dtw_path  # type: ignore

            def _dtw_func_tslearn(x: np.ndarray, y: np.ndarray) -> tuple:
                path_arr, dist = tslearn_dtw_path(
                    x.reshape(-1, 1), y.reshape(-1, 1)
                )
                path = [(int(p[0]), int(p[1])) for p in path_arr]
                return float(dist), path

            _dtw_func = _dtw_func_tslearn
        except ImportError:
            pass  # Use pure numpy fallback

    # Align all series to a common index
    asset_names = sorted(returns.keys())
    if len(asset_names) < 2:
        return pd.DataFrame()

    ret_df = pd.DataFrame(returns).sort_index().dropna(how="all")
    if ret_df.shape[0] < window + max_lag:
        return pd.DataFrame()

    dates = ret_df.index
    n_dates = len(dates)
    n_assets = len(asset_names)

    leader_scores = np.full((n_dates, n_assets), np.nan, dtype=float)
    follower_scores = np.full((n_dates, n_assets), np.nan, dtype=float)
    avg_lags = np.full((n_dates, n_assets), np.nan, dtype=float)

    vals = ret_df.values.astype(float)

    for t in range(window, n_dates):
        block = vals[t - window + 1: t + 1]

        # Skip if too many NaN
        valid_cols = np.isfinite(block).sum(axis=0) >= (window // 2)
        valid_idx = np.flatnonzero(valid_cols)
        if len(valid_idx) < 2:
            continue

        # For efficiency, use sub-sequences of length max_lag
        sub_len = min(max_lag, block.shape[0])
        sub_block = block[-sub_len:]

        leads_count = np.zeros(n_assets, dtype=float)
        follows_count = np.zeros(n_assets, dtype=float)
        lag_sum = np.zeros(n_assets, dtype=float)
        pair_count = np.zeros(n_assets, dtype=float)

        for ii in range(len(valid_idx)):
            for jj in range(ii + 1, len(valid_idx)):
                ci = valid_idx[ii]
                cj = valid_idx[jj]
                xi = sub_block[:, ci]
                xj = sub_block[:, cj]

                # Skip if either has NaN in sub-window
                if not (np.all(np.isfinite(xi)) and np.all(np.isfinite(xj))):
                    continue

                try:
                    _, path = _dtw_func(xi, xj)
                    lag = _dtw_avg_lag_from_path(path)
                except (ValueError, RuntimeError):
                    continue

                # Positive lag => xi leads xj
                if lag > 0:
                    leads_count[ci] += 1
                    follows_count[cj] += 1
                elif lag < 0:
                    leads_count[cj] += 1
                    follows_count[ci] += 1

                lag_sum[ci] += lag
                lag_sum[cj] -= lag
                pair_count[ci] += 1
                pair_count[cj] += 1

        total_pairs = max(float(pair_count.max()), 1.0)
        for ci in valid_idx:
            if pair_count[ci] > 0:
                leader_scores[t, ci] = leads_count[ci] / total_pairs
                follower_scores[t, ci] = follows_count[ci] / total_pairs
                avg_lags[t, ci] = lag_sum[ci] / pair_count[ci]

    # Build output MultiIndex DataFrame
    panel = []
    for i, name in enumerate(asset_names):
        f = pd.DataFrame(
            {
                "dtw_leader_score": leader_scores[:, i],
                "dtw_follower_score": follower_scores[:, i],
                "dtw_avg_lag": avg_lags[:, i],
            },
            index=dates,
        )
        f.index.name = "date"
        f["permno"] = name
        f = f.set_index("permno", append=True).reorder_levels([1, 0])
        panel.append(f)

    if not panel:
        return pd.DataFrame()
    out = pd.concat(panel).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 8) Path Signature Features
# ---------------------------------------------------------------------------

def _numpy_order2_signature(
    price_inc: np.ndarray,
    volume_inc: np.ndarray,
) -> np.ndarray:
    """
    Pure numpy computation of truncated order-2 path signature for a 2D path
    (price, volume).

    Order 1: [integral dp, integral dv]  (2 terms)
    Order 2: [integral dp dp, integral dp dv, integral dv dp, integral dv dv]  (4 terms)
    Total: 6 terms

    Uses the Chen identity for iterated integrals:
        S^{ij} = sum_{s<t} dx^i_s * dx^j_t
    which equals the area under the path for the (i,j) component.
    """
    n = len(price_inc)
    if n == 0:
        return np.full(6, np.nan)

    # Level 1: cumulative sums of increments
    s1_p = np.sum(price_inc)
    s1_v = np.sum(volume_inc)

    # Level 2: iterated integrals using Chen identity
    # S^{ij} = sum_{k=0}^{n-1} (cumsum_i[k]) * dj[k+1]
    # where cumsum_i is the running sum of di up to (but not including) step k+1
    cum_p = np.cumsum(price_inc)
    cum_v = np.cumsum(volume_inc)

    # For S^{ij} = sum_{s<t} di_s * dj_t
    # We need prefix sums: prefix[k] = sum of di[0..k-1]
    prefix_p = np.concatenate([[0.0], cum_p[:-1]])
    prefix_v = np.concatenate([[0.0], cum_v[:-1]])

    s2_pp = np.sum(prefix_p * price_inc)   # integral p dp
    s2_pv = np.sum(prefix_p * volume_inc)  # integral p dv
    s2_vp = np.sum(prefix_v * price_inc)   # integral v dp
    s2_vv = np.sum(prefix_v * volume_inc)  # integral v dv

    return np.array([s1_p, s1_v, s2_pp, s2_pv, s2_vp, s2_vv])


def compute_path_signatures(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    order: int = 2,
) -> pd.DataFrame:
    """
    Compute truncated path signatures of (price, volume) paths.

    For a 2D path the order-2 signature yields 6 terms:
        [integral dp, integral dv,
         integral p dp, integral p dv,
         integral v dp, integral v dv]

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'Close' and 'Volume' columns.
    windows : list of int
        Rolling window sizes (default [5, 20, 60]).
    order : int
        Signature truncation order (only 2 is supported in fallback).

    Returns
    -------
    pd.DataFrame
        Columns named ``sig_{w}d_{k}`` for window w and component k.
    """
    if windows is None:
        windows = [5, 20, 60]

    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    # Increments for the 2D path
    dp = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).values.astype(float)
    dv = np.log1p(volume).diff().replace([np.inf, -np.inf], np.nan).values.astype(float)

    n = len(df)
    n_sig_terms = 6  # order-2 for 2D path

    # Try iisignature for potentially higher-order / faster computation
    _use_iisig = False
    try:
        import iisignature  # type: ignore
        _use_iisig = True
    except ImportError:
        pass

    out = pd.DataFrame(index=df.index)

    for w in windows:
        minp = max(3, w // 3)
        sig_arr = np.full((n, n_sig_terms), np.nan, dtype=float)

        for end in range(w - 1, n):
            start = end - w + 1
            dp_win = dp[start: end + 1]
            dv_win = dv[start: end + 1]

            valid = np.isfinite(dp_win) & np.isfinite(dv_win)
            if valid.sum() < minp:
                continue

            dp_v = dp_win[valid]
            dv_v = dv_win[valid]

            if _use_iisig:
                try:
                    # iisignature expects a path (cumulative), not increments
                    path_2d = np.column_stack([
                        np.concatenate([[0.0], np.cumsum(dp_v)]),
                        np.concatenate([[0.0], np.cumsum(dv_v)]),
                    ])
                    sig = iisignature.sig(path_2d, order)
                    # iisignature returns order-1 (2 terms) + order-2 (4 terms) = 6 terms
                    sig_arr[end, :min(len(sig), n_sig_terms)] = sig[:n_sig_terms]
                except (ValueError, RuntimeError):
                    sig_arr[end] = _numpy_order2_signature(dp_v, dv_v)
            else:
                sig_arr[end] = _numpy_order2_signature(dp_v, dv_v)

        for k in range(n_sig_terms):
            out[f"sig_{w}d_{k}"] = sig_arr[:, k]

    return out.replace([np.inf, -np.inf], np.nan)
