"""
Regime detector with multiple engines and structural state layer.

Engines:
1) Rule-based thresholds (legacy, deterministic)
2) Probabilistic Gaussian HMM with sticky transitions + duration smoothing
3) Statistical Jump Model (PyPI or legacy)
4) Ensemble (majority vote across methods)

Structural State Layer (SPEC_03):
- BOCPD (Bayesian Online Change-Point Detection) for real-time changepoint signals
- ShockVector — unified, version-locked market state representation
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    BOCPD_ENABLED,
    BOCPD_HAZARD_LAMBDA,
    BOCPD_HAZARD_FUNCTION,
    BOCPD_RUNLENGTH_DEPTH,
    BOCPD_CHANGEPOINT_THRESHOLD,
    SHOCK_VECTOR_SCHEMA_VERSION,
    SHOCK_VECTOR_INCLUDE_STRUCTURAL,
)
from .hmm import (
    GaussianHMM,
    build_hmm_observation_matrix,
    map_raw_states_to_regimes,
    select_hmm_states_bic,
)
from .jump_model_legacy import StatisticalJumpModel
from .confidence_calibrator import ConfidenceCalibrator


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
    """Classifies market regime at each bar using rules, HMM, jump model, or ensemble.

    Optionally runs BOCPD (Bayesian Online Change-Point Detection) in parallel
    to provide real-time changepoint signals that complement the batch HMM.
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
        enable_bocpd: bool = BOCPD_ENABLED,
    ):
        """Initialize RegimeDetector.

        Parameters
        ----------
        method : str
            Detection method: ``"hmm"``, ``"jump"``, ``"rule"``, or ``"ensemble"``.
        enable_bocpd : bool
            If True, initialize a BOCPD detector for online changepoint signals.
        """
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

        # Confidence calibrator for weighted ensemble voting (SPEC_10)
        self._confidence_calibrator: Optional[ConfidenceCalibrator] = None

        # BOCPD: online changepoint detection (SPEC_03).
        self.enable_bocpd = enable_bocpd
        self._bocpd = None
        if enable_bocpd:
            try:
                from .bocpd import BOCPDDetector
                self._bocpd = BOCPDDetector(
                    hazard_lambda=BOCPD_HAZARD_LAMBDA,
                    hazard_func=BOCPD_HAZARD_FUNCTION,
                    max_runlength=BOCPD_RUNLENGTH_DEPTH,
                )
            except Exception as e:
                logger.warning("Failed to initialize BOCPD: %s", e)
                self.enable_bocpd = False

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
        """Confidence-weighted ensemble regime detection (SPEC_10 T2).

        Runs HMM and rule-based detection always, plus the statistical jump
        model when REGIME_JUMP_MODEL_ENABLED is True.  Each component
        contributes a weighted vote based on its confidence score and
        calibrated component weight.

        When no component achieves a weighted vote share above the
        disagreement threshold, the detector falls back to the configured
        uncertain regime (default: high_volatility / stress).
        """
        from ..config import (
            REGIME_ENSEMBLE_DEFAULT_WEIGHTS,
            REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD,
            REGIME_ENSEMBLE_UNCERTAIN_FALLBACK,
        )

        rule_out = self._rule_detect(features)
        hmm_out = self._hmm_detect(features)

        # Build the method list from genuinely independent detectors only.
        method_outputs: list[tuple[str, RegimeOutput]] = [
            ("rule", rule_out),
            ("hmm", hmm_out),
        ]
        if REGIME_JUMP_MODEL_ENABLED:
            method_outputs.append(("jump", self._jump_detect(features)))

        n_methods = len(method_outputs)
        T = len(features)

        # Get component weights (use calibrated if available, else defaults)
        if (
            hasattr(self, '_confidence_calibrator')
            and self._confidence_calibrator is not None
            and self._confidence_calibrator.fitted
        ):
            comp_weights = self._confidence_calibrator.component_weights
        else:
            comp_weights = dict(REGIME_ENSEMBLE_DEFAULT_WEIGHTS)

        # Normalize weights to components actually present
        present_weights = {
            name: comp_weights.get(name, 1.0 / n_methods)
            for name, _ in method_outputs
        }
        w_total = sum(present_weights.values())
        if w_total > 0:
            present_weights = {k: v / w_total for k, v in present_weights.items()}

        # Confidence-weighted voting
        consensus = np.full(T, 2, dtype=int)
        vote_confidence = np.full(T, 0.5, dtype=float)

        for t in range(T):
            # Accumulate weighted scores per regime
            regime_scores = np.zeros(4, dtype=float)
            for name, out in method_outputs:
                regime_t = int(out.regime.iloc[t])
                conf_t = float(out.confidence.iloc[t])
                weight = present_weights.get(name, 1.0 / n_methods)

                # Calibrate confidence if calibrator is available
                if (
                    hasattr(self, '_confidence_calibrator')
                    and self._confidence_calibrator is not None
                    and self._confidence_calibrator.fitted
                ):
                    conf_t = self._confidence_calibrator.calibrate(
                        conf_t, name, regime_t,
                    )

                if 0 <= regime_t <= 3:
                    regime_scores[regime_t] += weight * conf_t

            total_score = regime_scores.sum()
            if total_score > 0:
                best_regime = int(np.argmax(regime_scores))
                best_fraction = regime_scores[best_regime] / total_score

                if best_fraction >= REGIME_ENSEMBLE_DISAGREEMENT_THRESHOLD:
                    consensus[t] = best_regime
                    vote_confidence[t] = best_fraction
                else:
                    # Disagreement: fall back to uncertain regime
                    consensus[t] = REGIME_ENSEMBLE_UNCERTAIN_FALLBACK
                    vote_confidence[t] = best_fraction
            else:
                # All zero: use HMM as tiebreaker
                consensus[t] = int(hmm_out.regime.iloc[t])
                vote_confidence[t] = 1.0 / n_methods

        regime = pd.Series(consensus, index=features.index, dtype=int)

        # Blend probabilities using calibrated weights
        probs = pd.DataFrame(
            0.0, index=features.index,
            columns=[f"regime_prob_{i}" for i in range(4)],
        )
        for name, out in method_outputs:
            w = present_weights.get(name, 1.0 / n_methods)
            probs = probs.add(out.probabilities.fillna(0.0) * w, fill_value=0.0)

        probs = probs.fillna(0.0)
        row_sums = probs.sum(axis=1).replace(0, 1)
        probs = probs.div(row_sums, axis=0)
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

    def calibrate_confidence_weights(
        self,
        features: pd.DataFrame,
        actual_regimes: np.ndarray,
    ) -> Dict:
        """Calibrate ensemble confidence weights from validation data (SPEC_10 T2).

        Runs all ensemble components on the given features, collects their
        predictions and confidence scores, then fits the ECM calibrator.

        Parameters
        ----------
        features : pd.DataFrame
            Validation feature DataFrame.
        actual_regimes : np.ndarray
            True regime labels for the validation period.

        Returns
        -------
        dict
            Calibrated component weights.
        """
        rule_out = self._rule_detect(features)
        hmm_out = self._hmm_detect(features)

        predictions: Dict[str, np.ndarray] = {
            "rule": rule_out.confidence.values,
            "hmm": hmm_out.confidence.values,
        }
        predicted_regimes: Dict[str, np.ndarray] = {
            "rule": rule_out.regime.values,
            "hmm": hmm_out.regime.values,
        }

        if REGIME_JUMP_MODEL_ENABLED:
            jump_out = self._jump_detect(features)
            predictions["jump"] = jump_out.confidence.values
            predicted_regimes["jump"] = jump_out.regime.values

        calibrator = ConfidenceCalibrator(n_regimes=4)
        calibrator.fit(predictions, predicted_regimes, actual_regimes)
        self._confidence_calibrator = calibrator

        return calibrator.component_weights

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

    def detect_with_shock_context(
        self,
        features: pd.DataFrame,
        ticker: str = "",
    ) -> "ShockVector":
        """Detect regime and produce a ShockVector with BOCPD signals.

        Runs the standard regime detection pipeline plus optional BOCPD
        changepoint detection on the return series.  Returns a single
        ``ShockVector`` for the most recent bar.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame for a single security (rows = bars).
        ticker : str
            Security identifier for the ShockVector.

        Returns
        -------
        ShockVector
            Unified market state representation for the last bar.
        """
        from .shock_vector import ShockVector, ShockVectorValidator

        # Run standard regime detection.
        regime_out = self.detect_full(features)

        # BOCPD changepoint detection on return series.
        bocpd_cp_prob = 0.0
        bocpd_runlength = 0

        if self.enable_bocpd and self._bocpd is not None:
            returns = features.get(
                "return_1d",
                pd.Series(0.0, index=features.index),
            ).fillna(0.0).values

            if len(returns) >= 10:
                try:
                    batch_result = self._bocpd.batch_update(returns)
                    bocpd_cp_prob = float(batch_result.changepoint_probs[-1])
                    bocpd_runlength = int(batch_result.run_lengths[-1])
                except Exception as e:
                    logger.warning("BOCPD batch_update failed for %s: %s", ticker, e)

        # Jump detection: flag if the most recent return is a 2.5-sigma event.
        jump_detected = False
        jump_magnitude = 0.0
        ret_col = features.get("return_1d", pd.Series(0.0, index=features.index))
        ret_vals = ret_col.dropna().values
        if len(ret_vals) >= 20:
            recent_ret = float(ret_vals[-1])
            recent_vol = float(np.std(ret_vals[-20:]))
            if recent_vol > 1e-10:
                jump_detected = abs(recent_ret) > 2.5 * recent_vol
                jump_magnitude = recent_ret

        # Uncertainty from regime probabilities.
        uncertainty_series = regime_out.uncertainty
        hmm_uncertainty = 0.5
        if uncertainty_series is not None and len(uncertainty_series) > 0:
            hmm_uncertainty = float(uncertainty_series.iloc[-1])

        # Structural features (include if available).
        structural = {}
        if SHOCK_VECTOR_INCLUDE_STRUCTURAL:
            _structural_cols = {
                "spectral_entropy": "SpectralEntropy_252",
                "ssa_trend_strength": "SSATrendStr_60",
                "jump_intensity": "JumpIntensity_20",
                "eigenvalue_concentration": "EigenConcentration_60",
            }
            for key, col in _structural_cols.items():
                if col in features.columns:
                    val = features[col].iloc[-1]
                    if np.isfinite(val):
                        structural[key] = float(val)

        sv = ShockVector(
            schema_version=SHOCK_VECTOR_SCHEMA_VERSION,
            timestamp=features.index[-1] if hasattr(features.index[-1], 'isoformat') else pd.Timestamp.now(),
            ticker=ticker,
            hmm_regime=int(regime_out.regime.iloc[-1]),
            hmm_confidence=float(regime_out.confidence.iloc[-1]),
            hmm_uncertainty=hmm_uncertainty,
            bocpd_changepoint_prob=bocpd_cp_prob,
            bocpd_runlength=bocpd_runlength,
            jump_detected=jump_detected,
            jump_magnitude=jump_magnitude,
            structural_features=structural,
            transition_matrix=regime_out.transition_matrix,
            ensemble_model_type=regime_out.model_type,
        )

        is_valid, errors = ShockVectorValidator.validate(sv)
        if not is_valid:
            logger.error("Invalid ShockVector for %s: %s", ticker, errors)

        return sv

    def detect_batch_with_shock_context(
        self,
        features_by_id: Dict[str, pd.DataFrame],
    ) -> Dict[str, "ShockVector"]:
        """Detect regimes and produce ShockVectors for multiple securities.

        Parameters
        ----------
        features_by_id : dict
            Mapping from ticker/PERMNO to feature DataFrames.

        Returns
        -------
        dict[str, ShockVector]
            Mapping from ticker to ShockVector.
        """
        shock_vectors = {}
        for ticker, features in features_by_id.items():
            try:
                sv = self.detect_with_shock_context(features, ticker=str(ticker))
                shock_vectors[str(ticker)] = sv
            except Exception as e:
                logger.error("ShockVector generation failed for %s: %s", ticker, e)
        return shock_vectors

    @staticmethod
    def _get_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
        """Internal helper for get col."""
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series(default, index=df.index)


def validate_hmm_observation_features(features: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that required HMM observation features are present and non-degenerate.

    Checks the 4 core features (always required) and reports on the 7 extended
    features (optional but recommended).

    Parameters
    ----------
    features : pd.DataFrame
        Computed features from the feature pipeline.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, list_of_warnings)``.  ``is_valid`` is False only if
        core features are missing.
    """
    core_required = {
        "return_1d": "Core feature: 1-day return",
        "return_vol_20d": "Core feature: 20-day realized volatility",
        "NATR_14": "Core feature: Normalized Average True Range",
        "SMASlope_50": "Core feature: 50-day SMA slope",
    }
    extended_optional = {
        "GARCH_252": "Extended: GARCH volatility (for credit spread proxy)",
        "Volume": "Extended: Volume (for volume regime z-score)",
        "return_20d": "Extended: 20-day return (for momentum)",
        "Hurst_100": "Extended: Hurst exponent (for mean reversion signal)",
        "AutoCorr_20_1": "Extended: Autocorrelation (for cross-correlation proxy)",
    }

    warnings: List[str] = []
    core_missing = []

    for col, desc in core_required.items():
        if col not in features.columns:
            core_missing.append(col)
            warnings.append(f"MISSING {desc}: '{col}' not in features")
        else:
            nan_pct = features[col].isna().mean()
            if nan_pct > 0.10:
                warnings.append(
                    f"HIGH_NAN {desc}: '{col}' has {nan_pct:.1%} missing values"
                )

    for col, desc in extended_optional.items():
        if col not in features.columns:
            warnings.append(f"ABSENT {desc}: '{col}' not available (graceful fallback used)")

    is_valid = len(core_missing) == 0
    return is_valid, warnings


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

