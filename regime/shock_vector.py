"""
Unified Shock/Structure Vector — version-locked market state representation.

Combines HMM regime state, BOCPD online changepoint signals, jump detection,
and optional structural features into a single, validated dataclass that
downstream systems (backtester, models, dashboard) can consume.

Schema versioning ensures backward compatibility: V1 consumers can safely
ignore fields added in V2+.  Validation is enforced at construction time
and via the ``ShockVectorValidator`` for batch processing.

This module is part of the Structural State Layer (SPEC_03).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Supported schema versions — add new versions here as the schema evolves.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})


@dataclass
class ShockVector:
    """Unified market state and shock representation.

    Combines HMM regime, BOCPD online changepoint signals, jump detection,
    and optional structural features.  Version-locked for reproducibility
    and schema compatibility.

    Attributes
    ----------
    schema_version : str
        Version of the ShockVector schema (e.g., ``"1.0"``).
    timestamp : datetime
        Observation timestamp.
    ticker : str
        Security identifier.
    hmm_regime : int
        HMM-detected regime (0=trending_bull, 1=trending_bear,
        2=mean_reverting, 3=high_volatility).
    hmm_confidence : float
        HMM state confidence in [0, 1].
    hmm_uncertainty : float
        Normalized entropy of HMM posterior in [0, 1].
        High = uncertain, low = confident.
    bocpd_changepoint_prob : float
        BOCPD probability of a recent regime change in [0, 1].
    bocpd_runlength : int
        BOCPD most likely run-length (bars since last changepoint).
    jump_detected : bool
        True if a recent jump event (>= 2.5 sigma) was detected.
    jump_magnitude : float
        Magnitude of detected jump (percentage return).
    structural_features : dict
        Optional structural features keyed by name.
        E.g., ``{"spectral_entropy": 0.72, "ssa_trend_strength": 0.15}``.
    transition_matrix : np.ndarray or None
        HMM transition probability matrix, shape ``(4, 4)``.
        Not serialized to JSON.
    ensemble_model_type : str
        The model type that produced the regime label
        (``"hmm"``, ``"jump"``, ``"ensemble"``, ``"rule"``).
    """

    schema_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)
    ticker: str = ""
    hmm_regime: int = 0
    hmm_confidence: float = 0.5
    hmm_uncertainty: float = 0.5
    bocpd_changepoint_prob: float = 0.0
    bocpd_runlength: int = 0
    jump_detected: bool = False
    jump_magnitude: float = 0.0
    structural_features: Dict[str, float] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    ensemble_model_type: str = "hmm"

    def __post_init__(self) -> None:
        """Validate fields at construction time."""
        if self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema version: {self.schema_version!r}. "
                f"Supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if not (0 <= self.hmm_regime <= 3):
            raise ValueError(
                f"hmm_regime must be in [0, 3], got {self.hmm_regime}"
            )
        self.hmm_confidence = float(np.clip(self.hmm_confidence, 0.0, 1.0))
        self.hmm_uncertainty = float(np.clip(self.hmm_uncertainty, 0.0, 1.0))
        self.bocpd_changepoint_prob = float(
            np.clip(self.bocpd_changepoint_prob, 0.0, 1.0)
        )
        self.bocpd_runlength = max(0, int(self.bocpd_runlength))

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dictionary.

        Excludes numpy arrays (``transition_matrix``) and converts
        ``datetime`` to ISO 8601 string.
        """
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d.pop("transition_matrix", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ShockVector":
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with ShockVector fields.

        Returns
        -------
        ShockVector
        """
        d = dict(data)
        ts = d.get("timestamp")
        if isinstance(ts, str):
            d["timestamp"] = datetime.fromisoformat(ts)
        d.pop("transition_matrix", None)
        return cls(**d)

    def is_shock_event(self, changepoint_threshold: float = 0.50) -> bool:
        """Determine if the current vector represents a shock event.

        A shock is flagged when any of:
        - A jump was detected (>= 2.5 sigma)
        - BOCPD changepoint probability exceeds the threshold
        - The absolute jump magnitude exceeds 3%

        Parameters
        ----------
        changepoint_threshold : float
            Minimum BOCPD changepoint probability to flag a shock.

        Returns
        -------
        bool
        """
        return (
            self.jump_detected
            or self.bocpd_changepoint_prob > changepoint_threshold
            or abs(self.jump_magnitude) > 0.03
        )

    def regime_name(self) -> str:
        """Human-readable regime name using canonical REGIME_NAMES mapping."""
        _names = {
            0: "trending_bull",
            1: "trending_bear",
            2: "mean_reverting",
            3: "high_volatility",
        }
        return _names.get(self.hmm_regime, f"unknown_{self.hmm_regime}")


class ShockVectorValidator:
    """Validate ShockVector schema and data integrity.

    Provides both single-vector and batch validation with detailed
    error reporting.
    """

    @staticmethod
    def validate(sv: ShockVector) -> Tuple[bool, List[str]]:
        """Validate a single ShockVector.

        Parameters
        ----------
        sv : ShockVector
            The vector to validate.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_valid, list_of_error_messages)``.
        """
        errors: List[str] = []

        # Schema version
        if sv.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"Unsupported schema version: {sv.schema_version!r}"
            )

        # Regime
        if not isinstance(sv.hmm_regime, (int, np.integer)):
            errors.append(
                f"hmm_regime must be int, got {type(sv.hmm_regime).__name__}"
            )
        elif not (0 <= sv.hmm_regime <= 3):
            errors.append(
                f"hmm_regime must be in [0, 3], got {sv.hmm_regime}"
            )

        # Confidence bounds
        if not (0.0 <= sv.hmm_confidence <= 1.0):
            errors.append(
                f"hmm_confidence must be in [0, 1], got {sv.hmm_confidence}"
            )

        # Uncertainty bounds
        if not (0.0 <= sv.hmm_uncertainty <= 1.0):
            errors.append(
                f"hmm_uncertainty must be in [0, 1], got {sv.hmm_uncertainty}"
            )

        # BOCPD bounds
        if not (0.0 <= sv.bocpd_changepoint_prob <= 1.0):
            errors.append(
                f"bocpd_changepoint_prob must be in [0, 1], "
                f"got {sv.bocpd_changepoint_prob}"
            )

        if sv.bocpd_runlength < 0:
            errors.append(
                f"bocpd_runlength must be >= 0, got {sv.bocpd_runlength}"
            )

        # Ticker
        if not isinstance(sv.ticker, str):
            errors.append(
                f"ticker must be str, got {type(sv.ticker).__name__}"
            )

        # Structural features
        if not isinstance(sv.structural_features, dict):
            errors.append(
                f"structural_features must be dict, "
                f"got {type(sv.structural_features).__name__}"
            )
        else:
            for key, val in sv.structural_features.items():
                if not isinstance(key, str):
                    errors.append(
                        f"structural_features key {key!r} must be str"
                    )
                if not isinstance(val, (int, float, np.integer, np.floating)):
                    errors.append(
                        f"structural_features[{key!r}] must be numeric, "
                        f"got {type(val).__name__}"
                    )
                elif not np.isfinite(val):
                    errors.append(
                        f"structural_features[{key!r}] is not finite: {val}"
                    )

        # Transition matrix shape
        if sv.transition_matrix is not None:
            if not isinstance(sv.transition_matrix, np.ndarray):
                errors.append(
                    f"transition_matrix must be ndarray, "
                    f"got {type(sv.transition_matrix).__name__}"
                )
            elif sv.transition_matrix.shape != (4, 4):
                errors.append(
                    f"transition_matrix must be (4, 4), "
                    f"got {sv.transition_matrix.shape}"
                )

        # Model type
        valid_types = {"hmm", "jump", "ensemble", "rule"}
        if sv.ensemble_model_type not in valid_types:
            errors.append(
                f"ensemble_model_type must be one of {valid_types}, "
                f"got {sv.ensemble_model_type!r}"
            )

        return len(errors) == 0, errors

    @staticmethod
    def batch_validate(
        vectors: List[ShockVector],
    ) -> Dict[int, List[str]]:
        """Validate a batch of ShockVectors.

        Parameters
        ----------
        vectors : list[ShockVector]
            Vectors to validate.

        Returns
        -------
        dict[int, list[str]]
            Mapping from vector index to list of errors.
            Only invalid vectors are included.
        """
        errors_by_idx: Dict[int, List[str]] = {}
        for i, sv in enumerate(vectors):
            is_valid, errors = ShockVectorValidator.validate(sv)
            if not is_valid:
                errors_by_idx[i] = errors
        return errors_by_idx
