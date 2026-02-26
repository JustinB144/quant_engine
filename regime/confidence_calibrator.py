"""Confidence Calibrator for regime ensemble voting (SPEC_10 T2).

Implements Empirical Calibration Matrix (ECM) calibration so that each
ensemble component's reported confidence is mapped to realized frequency.
This corrects overconfident or underconfident detectors before combining
their votes.

Usage::

    calibrator = ConfidenceCalibrator(n_regimes=4)
    calibrator.fit(
        predictions={"hmm": hmm_confs, "rule": rule_confs, "jump": jump_confs},
        predicted_regimes={"hmm": hmm_regimes, "rule": rule_regimes, "jump": jump_regimes},
        actuals=actual_regimes,
    )
    calibrated = calibrator.calibrate(0.7, component="hmm", regime=1)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Empirical Calibration Matrix for regime detection confidence scores.

    For each (component, regime) pair, builds a mapping from raw confidence
    bins to realized frequency (how often the component is correct when it
    reports a given confidence level).

    Parameters
    ----------
    n_regimes : int
        Number of distinct regimes (default 4).
    n_bins : int
        Number of confidence bins for calibration curve (default 10).
    """

    def __init__(self, n_regimes: int = 4, n_bins: int = 10):
        self.n_regimes = n_regimes
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        # ECM: {component_name: np.ndarray of shape (n_regimes, n_bins)}
        # Each entry is the realized accuracy for that (component, regime, conf_bin).
        self._ecm: Dict[str, np.ndarray] = {}
        # Component weights calibrated from overall accuracy
        self._component_weights: Dict[str, float] = {}
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def component_weights(self) -> Dict[str, float]:
        """Return calibrated component weights (sum to 1)."""
        return dict(self._component_weights)

    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        predicted_regimes: Dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> "ConfidenceCalibrator":
        """Fit the calibration matrix from historical predictions.

        Parameters
        ----------
        predictions : dict
            ``{component_name: confidence_array}`` — 1-D arrays of confidence
            scores in [0, 1] for each timestep.
        predicted_regimes : dict
            ``{component_name: regime_array}`` — 1-D arrays of predicted regime
            labels (int, 0 to n_regimes-1) for each timestep.
        actuals : np.ndarray
            1-D array of true regime labels (int, 0 to n_regimes-1).

        Returns
        -------
        self
        """
        actuals = np.asarray(actuals, dtype=int)
        T = len(actuals)

        component_accuracies: Dict[str, float] = {}

        for comp_name in predictions:
            confs = np.asarray(predictions[comp_name], dtype=float)
            regimes = np.asarray(predicted_regimes[comp_name], dtype=int)

            if len(confs) != T or len(regimes) != T:
                raise ValueError(
                    f"Component '{comp_name}' array length mismatch: "
                    f"confs={len(confs)}, regimes={len(regimes)}, actuals={T}"
                )

            ecm = np.full((self.n_regimes, self.n_bins), 0.5)  # default: 50% accuracy
            counts = np.zeros((self.n_regimes, self.n_bins), dtype=int)

            for regime in range(self.n_regimes):
                for b in range(self.n_bins):
                    lo, hi = self.bin_edges[b], self.bin_edges[b + 1]
                    mask = (
                        (confs >= lo) & (confs < hi)
                        & (regimes == regime)
                    )
                    if b == self.n_bins - 1:
                        # Include right edge in last bin
                        mask = (
                            (confs >= lo) & (confs <= hi)
                            & (regimes == regime)
                        )
                    n_in_bin = mask.sum()
                    if n_in_bin >= 5:
                        correct = (actuals[mask] == regime).mean()
                        ecm[regime, b] = float(correct)
                        counts[regime, b] = n_in_bin

            self._ecm[comp_name] = ecm

            # Overall component accuracy
            correct_total = (regimes == actuals).sum()
            component_accuracies[comp_name] = correct_total / max(T, 1)

        # Compute weights proportional to accuracy
        total_acc = sum(component_accuracies.values())
        if total_acc > 0:
            self._component_weights = {
                comp: acc / total_acc
                for comp, acc in component_accuracies.items()
            }
        else:
            n_comp = len(predictions)
            self._component_weights = {
                comp: 1.0 / n_comp for comp in predictions
            }

        self._fitted = True
        logger.info(
            "ConfidenceCalibrator fitted on %d timesteps, %d components. "
            "Weights: %s",
            T, len(predictions),
            {k: f"{v:.3f}" for k, v in self._component_weights.items()},
        )
        return self

    def calibrate(
        self,
        confidence: float,
        component: str,
        regime: int,
    ) -> float:
        """Return calibrated confidence for a (component, regime, raw_conf) triple.

        If the calibrator is not fitted or the component is unknown, returns
        the raw confidence unchanged.

        Parameters
        ----------
        confidence : float
            Raw confidence score in [0, 1].
        component : str
            Name of the ensemble component (e.g., ``"hmm"``).
        regime : int
            Predicted regime label.

        Returns
        -------
        float
            Calibrated confidence in [0, 1].
        """
        if not self._fitted or component not in self._ecm:
            return float(confidence)

        regime = int(regime) % self.n_regimes
        ecm = self._ecm[component]

        # Find the bin for this confidence
        bin_idx = int(np.clip(
            np.digitize(confidence, self.bin_edges) - 1,
            0, self.n_bins - 1,
        ))

        return float(ecm[regime, bin_idx])

    def get_component_weight(self, component: str) -> float:
        """Return the calibrated weight for a component.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        float
            Weight in [0, 1] (all weights sum to 1).
        """
        return self._component_weights.get(component, 1.0 / max(len(self._component_weights), 1))

    def expected_calibration_error(
        self,
        predictions: Dict[str, np.ndarray],
        predicted_regimes: Dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Compute Expected Calibration Error (ECE) for each component.

        ECE measures the average gap between reported confidence and realized
        accuracy, weighted by the fraction of samples in each bin.

        Parameters
        ----------
        predictions, predicted_regimes, actuals : same as ``fit()``.

        Returns
        -------
        dict
            ``{component_name: ece_value}``.
        """
        actuals = np.asarray(actuals, dtype=int)
        T = len(actuals)
        ece_scores: Dict[str, float] = {}

        for comp_name in predictions:
            confs = np.asarray(predictions[comp_name], dtype=float)
            regimes = np.asarray(predicted_regimes[comp_name], dtype=int)

            ece = 0.0
            for b in range(self.n_bins):
                lo, hi = self.bin_edges[b], self.bin_edges[b + 1]
                if b == self.n_bins - 1:
                    mask = (confs >= lo) & (confs <= hi)
                else:
                    mask = (confs >= lo) & (confs < hi)
                n_in_bin = mask.sum()
                if n_in_bin == 0:
                    continue
                avg_conf = confs[mask].mean()
                accuracy = (regimes[mask] == actuals[mask]).mean()
                ece += (n_in_bin / T) * abs(avg_conf - accuracy)

            ece_scores[comp_name] = float(ece)

        return ece_scores
