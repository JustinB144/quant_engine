"""
Cross-market disagreement engine for Kalshi event features.

Standardized comparisons:
  - Kalshi distribution entropy vs options implied vol
  - Kalshi tail mass vs options skew
  - Gap z-scores and shock features
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass
class DisagreementSignals:
    """Container for cross-market disagreement features derived from Kalshi and options references."""
    entropy_gap: float
    entropy_gap_zscore: float
    tail_gap: float
    tail_gap_zscore: float
    iv_kalshi_vol_gap: float
    skew_kalshi_tail_gap: float


def compute_disagreement(
    kalshi_entropy: float,
    kalshi_tail_mass: float,
    kalshi_variance: float,
    options_iv: Optional[float] = None,
    options_skew: Optional[float] = None,
    entropy_history: Optional[pd.Series] = None,
    tail_history: Optional[pd.Series] = None,
) -> DisagreementSignals:
    """
    Compute cross-market disagreement signals.

    Args:
        kalshi_entropy: Shannon entropy of the Kalshi distribution snapshot.
        kalshi_tail_mass: Tail probability mass from Kalshi distribution.
        kalshi_variance: Variance of the Kalshi distribution.
        options_iv: ATM implied volatility from options market.
        options_skew: 25-delta skew from options market.
        entropy_history: Rolling series of past entropy values for z-scoring.
        tail_history: Rolling series of past tail mass values for z-scoring.
    """
    # Entropy gap: how much does Kalshi uncertainty differ from options?
    entropy_gap = np.nan
    if options_iv is not None and np.isfinite(options_iv) and options_iv > 0:
        implied_var = (options_iv ** 2) * (30.0 / 365.0)
        implied_entropy = 0.5 * np.log(2 * np.pi * np.e * max(implied_var, _EPS))
        entropy_gap = float(kalshi_entropy - implied_entropy)

    # Z-score the entropy gap against history
    entropy_gap_z = np.nan
    if entropy_history is not None and len(entropy_history) > 10:
        hist = entropy_history.dropna()
        if len(hist) > 10:
            mu = float(hist.mean())
            sigma = float(hist.std())
            if sigma > _EPS:
                entropy_gap_z = float((kalshi_entropy - mu) / sigma)

    # Tail gap: Kalshi tail mass vs options skew proxy
    tail_gap = np.nan
    skew_kalshi_tail = np.nan
    if options_skew is not None and np.isfinite(options_skew):
        # Sigmoid transform of skew to 0-1 range for comparability
        options_tail_proxy = 1.0 / (1.0 + np.exp(-options_skew / 0.10))
        tail_gap = float(kalshi_tail_mass - options_tail_proxy)
        skew_kalshi_tail = tail_gap

    tail_gap_z = np.nan
    if tail_history is not None and len(tail_history) > 10:
        hist = tail_history.dropna()
        if len(hist) > 10:
            mu = float(hist.mean())
            sigma = float(hist.std())
            if sigma > _EPS:
                tail_gap_z = float((kalshi_tail_mass - mu) / sigma)

    # IV-Kalshi vol gap
    iv_vol_gap = np.nan
    if options_iv is not None and np.isfinite(options_iv) and np.isfinite(kalshi_variance):
        kalshi_vol = float(np.sqrt(max(kalshi_variance, 0.0)))
        iv_vol_gap = float(options_iv - kalshi_vol)

    return DisagreementSignals(
        entropy_gap=float(entropy_gap) if np.isfinite(entropy_gap) else np.nan,
        entropy_gap_zscore=float(entropy_gap_z) if np.isfinite(entropy_gap_z) else np.nan,
        tail_gap=float(tail_gap) if np.isfinite(tail_gap) else np.nan,
        tail_gap_zscore=float(tail_gap_z) if np.isfinite(tail_gap_z) else np.nan,
        iv_kalshi_vol_gap=float(iv_vol_gap) if np.isfinite(iv_vol_gap) else np.nan,
        skew_kalshi_tail_gap=float(skew_kalshi_tail) if np.isfinite(skew_kalshi_tail) else np.nan,
    )


def disagreement_as_feature_dict(signals: DisagreementSignals) -> dict:
    """Convert disagreement signals to a flat dict for feature panel merging."""
    return {
        "disagree_entropy_gap": signals.entropy_gap,
        "disagree_entropy_gap_z": signals.entropy_gap_zscore,
        "disagree_tail_gap": signals.tail_gap,
        "disagree_tail_gap_z": signals.tail_gap_zscore,
        "disagree_iv_vol_gap": signals.iv_kalshi_vol_gap,
        "disagree_skew_tail_gap": signals.skew_kalshi_tail_gap,
    }
