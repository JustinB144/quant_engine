"""
Regime detector with two engines:
1) rule-based thresholds (legacy, deterministic)
2) probabilistic Gaussian HMM with sticky transitions + duration smoothing
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..config import (
    REGIME_MODEL_TYPE,
    REGIME_HMM_STATES,
    REGIME_HMM_MAX_ITER,
    REGIME_HMM_STICKINESS,
    REGIME_MIN_DURATION,
    REGIME_HMM_AUTO_SELECT_STATES,
    REGIME_HMM_MIN_STATES,
    REGIME_HMM_MAX_STATES,
)
from .hmm import (
    GaussianHMM,
    build_hmm_observation_matrix,
    map_raw_states_to_regimes,
    select_hmm_states_bic,
)


@dataclass
class RegimeOutput:
    """Unified regime detection output consumed by modeling, backtesting, and UI layers."""
    regime: pd.Series
    confidence: pd.Series
    probabilities: pd.DataFrame
    transition_matrix: Optional[np.ndarray]
    model_type: str


class RegimeDetector:
    """
    Classifies market regime at each bar using either rules or HMM.
    """

    def __init__(
        self,
        method: str = REGIME_MODEL_TYPE,
        hurst_trend_threshold: float = 0.55,
        hurst_mr_threshold: float = 0.45,
        adx_trend_threshold: float = 20,
        vol_spike_quantile: float = 0.80,
        vol_lookback: int = 252,
        hmm_states: int = REGIME_HMM_STATES,
        hmm_max_iter: int = REGIME_HMM_MAX_ITER,
        hmm_stickiness: float = REGIME_HMM_STICKINESS,
        min_duration: int = REGIME_MIN_DURATION,
    ):
        """Initialize RegimeDetector."""
        self.method = method
        self.hurst_trend = hurst_trend_threshold
        self.hurst_mr = hurst_mr_threshold
        self.adx_trend = adx_trend_threshold
        self.vol_spike_q = vol_spike_quantile
        self.vol_lookback = vol_lookback

        self.hmm_states = hmm_states
        self.hmm_max_iter = hmm_max_iter
        self.hmm_stickiness = hmm_stickiness
        self.min_duration = min_duration

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """detect."""
        return self.detect_with_confidence(features)[0]

    def _rule_detect(self, features: pd.DataFrame) -> RegimeOutput:
        """Internal helper for rule detect."""
        n = len(features)
        regime = pd.Series(np.full(n, 2), index=features.index, dtype=int)  # default: mean_reverting

        # Extract key features (with fallbacks)
        hurst = self._get_col(features, "Hurst_100", default=0.5)
        adx = self._get_col(features, "ADX_14", default=15)
        natr = self._get_col(features, "NATR_14", default=10)
        sma_slope = self._get_col(features, "SMASlope_50", default=0)

        vol_threshold = natr.rolling(self.vol_lookback, min_periods=60).quantile(self.vol_spike_q)
        high_vol = natr > vol_threshold
        trending = (hurst > self.hurst_trend) & (adx > self.adx_trend)
        bull_trend = trending & (sma_slope > 0)
        bear_trend = trending & (sma_slope <= 0)
        mean_rev = hurst < self.hurst_mr

        regime[mean_rev & ~high_vol] = 2
        regime[bull_trend & ~high_vol] = 0
        regime[bear_trend & ~high_vol] = 1
        regime[high_vol] = 3

        # Confidence from distance to decision boundaries.
        confidence = pd.Series(0.5, index=features.index)
        trending_mask = regime.isin([0, 1])
        confidence[trending_mask] = np.clip(0.5 + (hurst[trending_mask] - 0.5) * 2, 0.3, 1.0)
        mr_mask = regime == 2
        confidence[mr_mask] = np.clip(0.5 + (0.5 - hurst[mr_mask]) * 2, 0.3, 1.0)
        vol_mask = regime == 3
        vol_z = (natr - natr.rolling(252, min_periods=60).mean()) / (
            natr.rolling(252, min_periods=60).std() + 1e-10
        )
        confidence[vol_mask] = np.clip(0.5 + vol_z[vol_mask] * 0.2, 0.3, 1.0)

        probs = pd.DataFrame(index=features.index)
        for code in range(4):
            probs[f"regime_prob_{code}"] = (regime == code).astype(float)
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)

        return RegimeOutput(
            regime=regime.astype(int),
            confidence=confidence.astype(float),
            probabilities=probs.astype(float),
            transition_matrix=None,
            model_type="rule",
        )

    def _hmm_detect(self, features: pd.DataFrame) -> RegimeOutput:
        # If not enough data for robust HMM fit, fallback to deterministic rules.
        """Internal helper for hmm detect."""
        if len(features) < max(80, self.hmm_states * 16):
            return self._rule_detect(features)

        obs_df = build_hmm_observation_matrix(features)
        X = obs_df.values.astype(float)
        from ..config import REGIME_HMM_PRIOR_WEIGHT, REGIME_HMM_COVARIANCE_TYPE

        # Determine the number of states: use BIC selection if enabled, else use config.
        n_states = self.hmm_states
        if REGIME_HMM_AUTO_SELECT_STATES:
            try:
                n_states, _bic_scores = select_hmm_states_bic(
                    X,
                    min_states=REGIME_HMM_MIN_STATES,
                    max_states=REGIME_HMM_MAX_STATES,
                    max_iter=self.hmm_max_iter,
                    stickiness=self.hmm_stickiness,
                    min_duration=self.min_duration,
                    prior_weight=REGIME_HMM_PRIOR_WEIGHT,
                    covariance_type=REGIME_HMM_COVARIANCE_TYPE,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.warning("BIC state selection failed, falling back to %d states: %s", self.hmm_states, e)
                n_states = self.hmm_states

        model = GaussianHMM(
            n_states=n_states,
            max_iter=self.hmm_max_iter,
            stickiness=self.hmm_stickiness,
            min_duration=self.min_duration,
            prior_weight=REGIME_HMM_PRIOR_WEIGHT,
            covariance_type=REGIME_HMM_COVARIANCE_TYPE,
        )

        try:
            fit = model.fit(X)
            raw_states = fit.raw_states
            raw_probs = fit.state_probs
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return self._rule_detect(features)

        mapping = map_raw_states_to_regimes(raw_states, features)
        regime_vals = np.array([mapping.get(int(s), 2) for s in raw_states], dtype=int)
        regime = pd.Series(regime_vals, index=features.index, dtype=int)

        # Aggregate raw state probabilities into the canonical 4 regimes.
        probs = pd.DataFrame(0.0, index=features.index, columns=[f"regime_prob_{i}" for i in range(4)])
        for raw_s in range(raw_probs.shape[1]):
            reg = mapping.get(raw_s, 2)
            probs[f"regime_prob_{reg}"] += raw_probs[:, raw_s]
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
        confidence = probs.max(axis=1).clip(0.0, 1.0)

        return RegimeOutput(
            regime=regime,
            confidence=confidence.astype(float),
            probabilities=probs.astype(float),
            transition_matrix=fit.transition_matrix,
            model_type="hmm",
        )

    def detect_with_confidence(self, features: pd.DataFrame) -> tuple:
        """detect with confidence."""
        out = self.detect_full(features)
        return out.regime, out.confidence

    def detect_full(self, features: pd.DataFrame) -> RegimeOutput:
        """detect full."""
        if self.method == "hmm":
            return self._hmm_detect(features)
        return self._rule_detect(features)

    def regime_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime-derived features for the ML model.

        Returns DataFrame with:
            - regime: integer label
            - regime_confidence: 0-1 confidence
            - regime_duration: bars since last regime change
            - regime_prob_*: posterior regime probabilities
            - regime_*: one-hot encoded regimes
            - regime_model_type: "hmm" or "rule"
        """
        out = self.detect_full(features)
        regime = out.regime
        confidence = out.confidence

        result = pd.DataFrame(index=features.index)
        result["regime"] = regime
        result["regime_confidence"] = confidence
        result["regime_model_type"] = out.model_type

        # Duration since last regime change
        changes = regime.sort_index().diff().fillna(1).ne(0).astype(int)
        groups = changes.cumsum()
        result["regime_duration"] = groups.groupby(groups).cumcount() + 1

        # Posterior probabilities
        for col in out.probabilities.columns:
            result[col] = out.probabilities[col].astype(float)

        # One-hot encoding
        for code in range(4):
            result[f"regime_{code}"] = (regime == code).astype(float)

        # Transition probabilities from previous regime if available
        if out.transition_matrix is not None and out.transition_matrix.shape[0] >= 4:
            prev_regime = regime.shift(1).fillna(regime.iloc[0]).astype(int)
            trans_conf = np.zeros(len(regime), dtype=float)
            for i, curr in enumerate(regime.values):
                prev = int(prev_regime.iloc[i])
                if 0 <= prev < out.transition_matrix.shape[0] and 0 <= curr < out.transition_matrix.shape[1]:
                    trans_conf[i] = float(out.transition_matrix[prev, curr])
                else:
                    trans_conf[i] = 0.0
            result["regime_transition_prob"] = trans_conf
        else:
            result["regime_transition_prob"] = 0.0

        return result

    @staticmethod
    def _get_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
        """Internal helper for get col."""
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series(default, index=df.index)


def detect_regimes_batch(
    features_by_id: Dict[str, pd.DataFrame],
    detector: Optional[RegimeDetector] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    Shared regime detection across multiple PERMNOs.

    Replaces the duplicated regime-detection loop in run_train, run_backtest,
    run_retrain. Returns (regime_data, regimes_series, regime_probs_or_none).
    """
    det = detector or RegimeDetector()
    regime_dfs = []
    regime_prob_dfs = []

    for permno, feats in features_by_id.items():
        regime_df = det.regime_features(feats)
        regime_df["permno"] = permno
        regime_df = regime_df.set_index("permno", append=True).reorder_levels([1, 0])
        regime_dfs.append(regime_df)
        prob_cols = [c for c in regime_df.columns if c.startswith("regime_prob_")]
        if prob_cols:
            regime_prob_dfs.append(regime_df[prob_cols])

    if not regime_dfs:
        empty = pd.DataFrame()
        return empty, pd.Series(dtype=int), None

    regime_data = pd.concat(regime_dfs)
    regimes = regime_data["regime"]
    regime_probs = pd.concat(regime_prob_dfs) if regime_prob_dfs else None

    if verbose:
        from ..config import REGIME_NAMES
        counts = regimes.value_counts().sort_index()
        for code, count in counts.items():
            name = REGIME_NAMES.get(code, f"regime_{code}")
            print(f"    {name}: {count}")

    return regime_data, regimes, regime_probs

