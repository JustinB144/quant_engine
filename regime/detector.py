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
    REGIME_JUMP_MODEL_ENABLED,
    REGIME_JUMP_PENALTY,
    REGIME_EXPECTED_CHANGES_PER_YEAR,
    REGIME_ENSEMBLE_ENABLED,
    REGIME_ENSEMBLE_CONSENSUS_THRESHOLD,
)
from .hmm import (
    GaussianHMM,
    build_hmm_observation_matrix,
    map_raw_states_to_regimes,
    select_hmm_states_bic,
)
from .jump_model_legacy import StatisticalJumpModel


@dataclass
class RegimeOutput:
    """Unified regime detection output consumed by modeling, backtesting, and UI layers."""
    regime: pd.Series
    confidence: pd.Series
    probabilities: pd.DataFrame
    transition_matrix: Optional[np.ndarray]
    model_type: str
    uncertainty: Optional[pd.Series] = None  # Entropy of posterior probabilities


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

    def _apply_min_duration(
        self, regime: pd.Series, confidence: pd.Series,
    ) -> pd.Series:
        """Merge regime runs shorter than ``min_duration`` into adjacent regimes.

        Matches the HSMM-like smoothing in ``GaussianHMM._smooth_duration``
        but operates on ``pd.Series`` instead of ``np.ndarray``.  For each
        short run, the side (left or right neighbor) with higher mean
        confidence on the short segment wins.
        """
        if len(regime) == 0 or self.min_duration <= 1:
            return regime

        vals = regime.values.copy()
        conf = confidence.values
        n = len(vals)

        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[j] == vals[i]:
                j += 1
            run_len = j - i
            if run_len < self.min_duration:
                left_state = int(vals[i - 1]) if i > 0 else None
                right_state = int(vals[j]) if j < n else None

                if left_state is None and right_state is None:
                    i = j
                    continue
                if left_state is None:
                    repl = right_state
                elif right_state is None:
                    repl = left_state
                else:
                    left_score = float(conf[i:j].mean()) if left_state == int(vals[max(0, i - 1)]) else 0.0
                    right_score = float(conf[i:j].mean()) if right_state == int(vals[min(j, n - 1)]) else 0.0
                    # Prefer the neighbor whose confidence for the short segment is higher.
                    # When tied, prefer left (earlier regime persists).
                    repl = left_state if left_score >= right_score else right_state

                vals[i:j] = repl
            i = j

        return pd.Series(vals, index=regime.index, dtype=int)

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

        # Enforce minimum regime duration to prevent 1-bar flickers.
        if self.min_duration > 1:
            regime = self._apply_min_duration(regime, confidence)

        probs = pd.DataFrame(index=features.index)
        for code in range(4):
            probs[f"regime_prob_{code}"] = (regime == code).astype(float)
        probs = probs.fillna(0.0)
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
        probs = probs.fillna(0.0)
        confidence = confidence.fillna(0.5)

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
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                "HMM fit failed, falling back to rule-based detection: %s | n_samples=%d | n_states=%d",
                e, len(X), n_states,
            )
            return self._rule_detect(features)

        mapping = map_raw_states_to_regimes(raw_states, features)
        regime_vals = np.array([mapping.get(int(s), 2) for s in raw_states], dtype=int)
        regime = pd.Series(regime_vals, index=features.index, dtype=int)

        # Aggregate raw state probabilities into the canonical 4 regimes.
        probs = pd.DataFrame(0.0, index=features.index, columns=[f"regime_prob_{i}" for i in range(4)])
        for raw_s in range(raw_probs.shape[1]):
            reg = mapping.get(raw_s, 2)
            col = f"regime_prob_{reg}"
            probs.loc[:, col] = probs[col].values + raw_probs[:, raw_s]
        probs = probs.fillna(0.0)  # Guard against NaN from HMM posteriors
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
        probs = probs.fillna(0.0)  # Guard against NaN after normalization
        # If any row sums to 0 (all NaN before fillna), assign uniform distribution
        zero_rows = probs.sum(axis=1) == 0
        if zero_rows.any():
            probs.loc[zero_rows] = 0.25
        confidence = probs.max(axis=1).clip(0.0, 1.0)

        # Compute regime uncertainty (entropy of posterior)
        uncertainty = self.get_regime_uncertainty(probs)

        return RegimeOutput(
            regime=regime,
            confidence=confidence.astype(float),
            probabilities=probs.astype(float),
            transition_matrix=fit.transition_matrix,
            model_type="hmm",
            uncertainty=uncertainty,
        )

    def _jump_detect(self, features: pd.DataFrame) -> RegimeOutput:
        """Detect regimes using the Statistical Jump Model.

        When ``REGIME_JUMP_USE_PYPI_PACKAGE`` is True (default), uses the
        ``jumpmodels`` PyPI package via :class:`PyPIJumpModel`.  Otherwise
        falls back to the legacy in-repo ``StatisticalJumpModel``.
        """
        if len(features) < 80:
            return self._rule_detect(features)

        obs_df = build_hmm_observation_matrix(features)
        X = obs_df.values.astype(float)

        from ..config import REGIME_JUMP_USE_PYPI_PACKAGE

        try:
            if REGIME_JUMP_USE_PYPI_PACKAGE:
                result = self._jump_detect_pypi(X, features)
            else:
                result = self._jump_detect_legacy(X)
        except (ValueError, RuntimeError, ImportError) as e:
            logger.warning("Jump model fit failed, falling back to rules: %s", e)
            return self._rule_detect(features)

        # Map raw states to semantic regimes using the same approach as HMM
        mapping = map_raw_states_to_regimes(result.regime_sequence, features)
        regime_vals = np.array(
            [mapping.get(int(s), 2) for s in result.regime_sequence], dtype=int
        )
        regime = pd.Series(regime_vals, index=features.index, dtype=int)

        # Build canonical 4-regime probability matrix
        probs = pd.DataFrame(
            0.0, index=features.index, columns=[f"regime_prob_{i}" for i in range(4)]
        )
        for raw_s in range(result.regime_probs.shape[1]):
            reg = mapping.get(raw_s, 2)
            col = f"regime_prob_{reg}"
            probs.loc[:, col] = probs[col].values + result.regime_probs[:, raw_s]
        probs = probs.fillna(0.0)  # Guard against NaN from jump model posteriors
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
        probs = probs.fillna(0.0)  # Guard against NaN after normalization
        # If any row sums to 0 (all NaN before fillna), assign uniform distribution
        zero_rows = probs.sum(axis=1) == 0
        if zero_rows.any():
            probs.loc[zero_rows] = 0.25
        confidence = probs.max(axis=1).clip(0.0, 1.0)
        uncertainty = self.get_regime_uncertainty(probs)

        return RegimeOutput(
            regime=regime,
            confidence=confidence.astype(float),
            probabilities=probs.astype(float),
            transition_matrix=None,
            model_type="jump",
            uncertainty=uncertainty,
        )

    def _jump_detect_pypi(self, X: np.ndarray, features: pd.DataFrame):
        """Fit the PyPI JumpModel wrapper and return a JumpModelResult."""
        from .jump_model_pypi import PyPIJumpModel

        model = PyPIJumpModel(
            n_regimes=min(self.hmm_states, 4),
        )
        return model.fit(X)

    def _jump_detect_legacy(self, X: np.ndarray):
        """Fit the legacy StatisticalJumpModel and return a JumpModelResult."""
        jump_penalty = StatisticalJumpModel.compute_jump_penalty_from_data(
            REGIME_EXPECTED_CHANGES_PER_YEAR
        )
        model = StatisticalJumpModel(
            n_regimes=min(self.hmm_states, 4),
            jump_penalty=jump_penalty,
            max_iter=50,
        )
        return model.fit(X)

    def detect_ensemble(self, features: pd.DataFrame) -> RegimeOutput:
        """Ensemble regime detection using only genuinely independent methods.

        Runs HMM and rule-based detection always, plus the statistical jump
        model when REGIME_JUMP_MODEL_ENABLED is True.  Only declares a regime
        change when at least ``REGIME_ENSEMBLE_CONSENSUS_THRESHOLD`` methods
        agree (capped at ``n_methods``), reducing false switches.

        With 2 methods the threshold becomes ``min(threshold, 2)`` which
        requires unanimity — no phantom third vote inflates agreement.
        """
        rule_out = self._rule_detect(features)
        hmm_out = self._hmm_detect(features)

        # Build the method list from genuinely independent detectors only.
        method_outputs = [("rule", rule_out), ("hmm", hmm_out)]
        if REGIME_JUMP_MODEL_ENABLED:
            method_outputs.append(("jump", self._jump_detect(features)))

        n_methods = len(method_outputs)
        threshold = min(REGIME_ENSEMBLE_CONSENSUS_THRESHOLD, n_methods)

        # Stack regime arrays: (T, n_methods)
        methods = np.column_stack([m.regime.values for _, m in method_outputs])

        # Per-timestep majority vote
        T = len(features)
        consensus = np.full(T, 2, dtype=int)  # default: mean_reverting
        vote_confidence = np.full(T, 0.5, dtype=float)

        for t in range(T):
            votes = methods[t]
            unique, counts = np.unique(votes, return_counts=True)
            best_idx = np.argmax(counts)
            if counts[best_idx] >= threshold:
                consensus[t] = unique[best_idx]
                vote_confidence[t] = counts[best_idx] / n_methods
            else:
                # No consensus — use HMM prediction as tiebreaker (most sophisticated)
                consensus[t] = hmm_out.regime.iloc[t]
                vote_confidence[t] = 1.0 / n_methods

        regime = pd.Series(consensus, index=features.index, dtype=int)

        # Blend probabilities: equal weights across available methods
        weight = 1.0 / n_methods
        probs = sum(
            m.probabilities.fillna(0.0) * weight for _, m in method_outputs
        )
        probs = probs.fillna(0.0)
        probs = probs.div(probs.sum(axis=1).replace(0, 1), axis=0)
        probs = probs.fillna(0.0)

        confidence = pd.Series(vote_confidence, index=features.index, dtype=float)
        uncertainty = self.get_regime_uncertainty(probs)

        return RegimeOutput(
            regime=regime,
            confidence=confidence,
            probabilities=probs.astype(float),
            transition_matrix=hmm_out.transition_matrix,
            model_type="ensemble",
            uncertainty=uncertainty,
        )

    def detect_with_confidence(self, features: pd.DataFrame) -> tuple:
        """detect with confidence."""
        out = self.detect_full(features)
        return out.regime, out.confidence

    def detect_full(self, features: pd.DataFrame) -> RegimeOutput:
        """detect full."""
        if REGIME_ENSEMBLE_ENABLED and self.method == "hmm":
            return self.detect_ensemble(features)
        if self.method == "hmm":
            return self._hmm_detect(features)
        if self.method == "jump":
            return self._jump_detect(features)
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
    def get_regime_uncertainty(probabilities: pd.DataFrame) -> pd.Series:
        """Compute entropy of posterior regime probabilities as uncertainty measure.

        Entropy = -sum(p * log(p)) for each time step.
        High entropy = uncertain about regime. Low entropy = confident.
        Normalized to [0, 1] by dividing by max possible entropy (log(n_regimes)).
        """
        prob_cols = [c for c in probabilities.columns if c.startswith("regime_prob_")]
        if not prob_cols:
            return pd.Series(1.0, index=probabilities.index)
        probs = probabilities[prob_cols].clip(lower=1e-10).values
        n_regimes = probs.shape[1]
        entropy = -np.sum(probs * np.log(probs), axis=1)
        max_entropy = np.log(n_regimes) if n_regimes > 1 else 1.0
        return pd.Series(entropy / max_entropy, index=probabilities.index, dtype=float)

    @staticmethod
    def map_raw_states_to_regimes_stable(
        raw_states: np.ndarray,
        features: pd.DataFrame,
        reference_distributions: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> Dict[int, int]:
        """Map HMM states to semantic regimes using Wasserstein distance matching.

        Unlike the simple heuristic mapping, this method computes feature statistics
        per HMM state and matches them to reference distributions. This prevents
        label swaps across retrains by anchoring to stable reference profiles.

        Args:
            raw_states: HMM state labels
            features: feature DataFrame for computing per-state stats
            reference_distributions: optional dict {regime_code: {"mean_ret", "mean_vol"}}
                If None, uses built-in defaults.
        """
        if reference_distributions is None:
            reference_distributions = {
                0: {"mean_ret": 0.001, "mean_vol": 12.0},   # trending_bull
                1: {"mean_ret": -0.001, "mean_vol": 14.0},  # trending_bear
                2: {"mean_ret": 0.0, "mean_vol": 10.0},     # mean_reverting
                3: {"mean_ret": 0.0, "mean_vol": 25.0},     # high_volatility
            }

        unique_states = sorted(set(raw_states))
        state_profiles: Dict[int, Dict[str, float]] = {}
        for s in unique_states:
            mask = raw_states == s
            if mask.sum() == 0:
                continue
            ret = features.get("return_1d", pd.Series(0.0, index=features.index))[mask].mean()
            vol = features.get("NATR_14", pd.Series(10.0, index=features.index))[mask].mean()
            state_profiles[int(s)] = {"mean_ret": float(ret), "mean_vol": float(vol)}

        if not state_profiles:
            return {0: 2}

        # Compute distance from each state to each reference regime
        mapping: Dict[int, int] = {}
        assigned_regimes = set()

        # Sort regimes by distinctiveness (high vol first, then by return magnitude)
        regime_order = sorted(
            reference_distributions.keys(),
            key=lambda r: reference_distributions[r]["mean_vol"],
            reverse=True,
        )

        for regime_code in regime_order:
            ref = reference_distributions[regime_code]
            best_state = None
            best_dist = float("inf")
            for s, prof in state_profiles.items():
                if s in mapping:
                    continue
                # Wasserstein-like L1 distance (normalized)
                dist = (
                    abs(prof["mean_ret"] - ref["mean_ret"]) * 1000  # Scale returns
                    + abs(prof["mean_vol"] - ref["mean_vol"])
                )
                if dist < best_dist:
                    best_dist = dist
                    best_state = s
            if best_state is not None:
                mapping[best_state] = regime_code

        # Assign any remaining unmapped states to mean_reverting
        for s in state_profiles:
            if s not in mapping:
                mapping[s] = 2

        return mapping

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

