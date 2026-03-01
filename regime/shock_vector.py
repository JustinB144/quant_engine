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
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
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
    n_hmm_states: int = 4  # Actual number of HMM states used

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

    @classmethod
    def empty(cls, ticker: str = "") -> "ShockVector":
        """Return a default/neutral ShockVector marked as invalid.

        Used when inputs are empty or detection cannot proceed.
        """
        return cls(
            ticker=ticker,
            hmm_confidence=0.0,
            hmm_uncertainty=1.0,
            ensemble_model_type="rule",
        )

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
        # Filter to known fields for forward compatibility (V1 ignores V2+ fields)
        known = {f.name for f in dataclass_fields(cls)}
        unknown = set(d.keys()) - known
        if unknown:
            logger.debug(
                "ShockVector.from_dict ignoring unknown keys (possible newer schema): %s",
                unknown,
            )
        d = {k: v for k, v in d.items() if k in known}
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

        # Transition matrix shape — accept variable-sized square matrices
        if sv.transition_matrix is not None:
            from ..config_structured import MAX_HMM_STATES

            if not isinstance(sv.transition_matrix, np.ndarray):
                errors.append(
                    f"transition_matrix must be ndarray, "
                    f"got {type(sv.transition_matrix).__name__}"
                )
            else:
                tm = np.asarray(sv.transition_matrix)
                if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
                    errors.append(
                        f"Transition matrix must be square, got shape {tm.shape}"
                    )
                elif tm.shape[0] < 2 or tm.shape[0] > MAX_HMM_STATES:
                    errors.append(
                        f"Transition matrix size {tm.shape[0]} outside valid range "
                        f"[2, {MAX_HMM_STATES}]"
                    )

        # Model type
        valid_types = {"hmm", "jump", "ensemble", "rule"}
        if sv.ensemble_model_type not in valid_types:
            errors.append(
                f"ensemble_model_type must be one of {valid_types}, "
                f"got {sv.ensemble_model_type!r}"
            )

        # Timestamp type check (SPEC_AUDIT_FIX_27 T4)
        if not isinstance(sv.timestamp, datetime):
            errors.append(
                f"timestamp must be datetime, got {type(sv.timestamp).__name__}"
            )

        # Jump detected type check (SPEC_AUDIT_FIX_27 T4)
        if not isinstance(sv.jump_detected, bool):
            errors.append(
                f"jump_detected must be bool, got {type(sv.jump_detected).__name__}: {sv.jump_detected}"
            )

        # Jump magnitude type and range check (SPEC_AUDIT_FIX_27 T4)
        if not isinstance(sv.jump_magnitude, (int, float)):
            errors.append(
                f"jump_magnitude must be numeric, got {type(sv.jump_magnitude).__name__}"
            )
        elif not np.isfinite(sv.jump_magnitude):
            errors.append(f"jump_magnitude must be finite, got {sv.jump_magnitude}")

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


# ── Structural Feature Builder (SPEC_AUDIT_FIX_27 T2) ───────────────────


# Feature column mapping for structural features extracted from a feature DataFrame.
_STRUCTURAL_FEATURE_COLS = {
    "spectral_entropy": "SpectralEntropy_252",
    "ssa_trend_strength": "SSATrendStr_60",
    "jump_intensity": "JumpIntensity_20",
    "eigenvalue_concentration": "EigenConcentration_60",
}


def _build_structural_features(
    drift_score: float = 0.0,
    systemic_stress: float = 0.0,
    features: Optional["pd.DataFrame"] = None,
    bar_idx: int = -1,
) -> Dict[str, float]:
    """Build a unified structural feature dict for a single bar.

    Computes drift_score and systemic_stress from caller-provided values.
    If a feature DataFrame is provided, also extracts spectral_entropy,
    ssa_trend_strength, jump_intensity, and eigenvalue_concentration.

    Parameters
    ----------
    drift_score : float
        Trend conviction score in [0, 1].
    systemic_stress : float
        Realized vol percentile in [0, 1].
    features : pd.DataFrame, optional
        Feature DataFrame for extracting structural feature columns.
    bar_idx : int
        Row index into ``features`` for single-bar extraction (default -1 = last bar).

    Returns
    -------
    dict[str, float]
        Structural features with consistent keys across construction paths.
    """
    result: Dict[str, float] = {
        "drift_score": float(drift_score),
        "systemic_stress": float(systemic_stress),
    }

    # Feature-derived features (available when feature DataFrame is provided)
    if features is not None:
        for key, col in _STRUCTURAL_FEATURE_COLS.items():
            if col in features.columns:
                val = features[col].iloc[bar_idx]
                if np.isfinite(val):
                    result[key] = float(val)

    return result


# ── Batch ShockVector Computation (SPEC-W01) ────────────────────────────


def compute_shock_vectors(
    ohlcv: "pd.DataFrame",
    regime_series: Optional["pd.Series"] = None,
    regime_confidence_series: Optional["pd.Series"] = None,
    ticker: str = "",
    bocpd_hazard_lambda: float = 1.0 / 60,
    bocpd_hazard_func: str = "constant",
    bocpd_max_runlength: int = 200,
    jump_sigma_threshold: float = 2.5,
    vol_lookback: int = 20,
    changepoint_threshold: Optional[float] = None,
    ensemble_model_type: str = "hmm",
    features: Optional["pd.DataFrame"] = None,
) -> Dict:
    """Compute ShockVectors for every bar in a price series.

    Pre-computes structural state (BOCPD changepoint probability, jump
    detection, regime uncertainty) for each bar, enabling downstream systems
    (backtester, execution simulator) to condition on structural state
    without re-running full regime detection.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV DataFrame for a single security.  Must have a ``Close``
        column and a DatetimeIndex (or similar ordered index).
    regime_series : pd.Series, optional
        Per-bar regime labels (0-3), aligned to ``ohlcv.index``.
        If ``None``, defaults to regime 0 for all bars.
    regime_confidence_series : pd.Series, optional
        Per-bar **regime detection** confidence in [0, 1], aligned to
        ``ohlcv.index``. This is the maximum posterior probability across
        regime states (from RegimeDetector), NOT model prediction confidence.
        If ``None``, defaults to 0.5 (maximum uncertainty).
    ticker : str
        Security identifier embedded in each ShockVector.
    bocpd_hazard_lambda : float
        BOCPD constant hazard rate (expected changepoints per bar).
    bocpd_hazard_func : str
        BOCPD hazard function type: ``"constant"`` or ``"geometric"``.
    bocpd_max_runlength : int
        Maximum run-length tracked by the BOCPD detector.
    jump_sigma_threshold : float
        Sigma threshold for jump detection (default 2.5).
    vol_lookback : int
        Lookback window for realized volatility and jump detection.
    changepoint_threshold : float, optional
        BOCPD changepoint probability threshold for shock event detection.
        If ``None``, defaults to ``BOCPD_CHANGEPOINT_THRESHOLD`` from config.
    ensemble_model_type : str
        Model type that produced the regime labels (``"hmm"``, ``"jump"``,
        ``"ensemble"``, ``"rule"``).  Default ``"hmm"``.
    features : pd.DataFrame, optional
        Feature DataFrame aligned to ``ohlcv`` for extracting structural
        feature columns (spectral_entropy, ssa_trend_strength, etc.).

    Returns
    -------
    dict
        Mapping from index value (typically ``pd.Timestamp``) to
        ``ShockVector``.  One entry per bar in ``ohlcv``.
    """
    import pandas as pd

    # Resolve config-driven defaults
    from ..config import SHOCK_VECTOR_SCHEMA_VERSION
    if changepoint_threshold is None:
        try:
            from ..config import BOCPD_CHANGEPOINT_THRESHOLD
            changepoint_threshold = BOCPD_CHANGEPOINT_THRESHOLD
        except ImportError:
            changepoint_threshold = 0.50

    if ohlcv is None or len(ohlcv) == 0:
        return {}

    if "Close" not in ohlcv.columns:
        logger.warning("compute_shock_vectors: 'Close' column missing; returning empty dict")
        return {}

    close = ohlcv["Close"].astype(float)
    n = len(close)

    # ── Compute daily returns ──
    returns = close.pct_change().fillna(0.0).values

    # ── BOCPD batch processing ──
    bocpd_cp_probs = np.zeros(n)
    bocpd_runlengths = np.zeros(n, dtype=int)

    try:
        from .bocpd import BOCPDDetector

        bocpd = BOCPDDetector(
            hazard_lambda=bocpd_hazard_lambda,
            hazard_func=bocpd_hazard_func,
            max_runlength=bocpd_max_runlength,
        )
        batch_result = bocpd.batch_update(returns)
        bocpd_cp_probs = batch_result.changepoint_probs
        bocpd_runlengths = batch_result.run_lengths
    except Exception as e:
        logger.warning(
            "compute_shock_vectors: BOCPD batch processing failed for %s: %s",
            ticker, e,
        )

    # ── Per-bar jump detection ──
    jump_flags = np.zeros(n, dtype=bool)
    jump_magnitudes = np.zeros(n)

    for i in range(vol_lookback, n):
        window = returns[i - vol_lookback : i]
        window_std = float(np.std(window))
        if window_std > 1e-10:
            current_ret = returns[i]
            if abs(current_ret) > jump_sigma_threshold * window_std:
                jump_flags[i] = True
            jump_magnitudes[i] = current_ret

    # ── Per-bar uncertainty from regime confidence ──
    # Uncertainty ≈ 1 - confidence.  When the regime detector is confident
    # (high probability on one state), uncertainty is low.  This is a
    # monotone proxy for the full entropy computation.
    if regime_confidence_series is not None and len(regime_confidence_series) > 0:
        # Warn if values look like model confidence (typically > 0.8 for most bars)
        # vs regime confidence (typically 0.4-0.9 with more variance)
        mean_conf = float(regime_confidence_series.mean())
        if mean_conf > 0.95:
            logger.warning(
                "regime_confidence_series mean=%.3f is suspiciously high — "
                "verify this is regime confidence, not model prediction confidence",
                mean_conf,
            )
        conf_vals = regime_confidence_series.reindex(ohlcv.index).fillna(0.5).values
        conf_vals = np.clip(conf_vals.astype(float), 0.0, 1.0)
        uncertainty_vals = 1.0 - conf_vals
    else:
        conf_vals = np.full(n, 0.5)
        uncertainty_vals = np.full(n, 0.5)

    # ── Per-bar regime ──
    if regime_series is not None and len(regime_series) > 0:
        regime_vals = regime_series.reindex(ohlcv.index).fillna(0).values
        regime_vals = np.clip(regime_vals.astype(int), 0, 3)
    else:
        regime_vals = np.zeros(n, dtype=int)

    # ── Per-bar drift score (trend conviction) ──
    # drift_score = |price - SMA| / (ATR * sqrt(lookback)), clipped to [0, 1].
    # High drift = strong trend = lower execution urgency.
    drift_scores = np.zeros(n)
    if n >= vol_lookback:
        sma = pd.Series(close.values).rolling(vol_lookback, min_periods=1).mean().values
        high_vals = ohlcv["High"].values.astype(float) if "High" in ohlcv.columns else close.values
        low_vals = ohlcv["Low"].values.astype(float) if "Low" in ohlcv.columns else close.values
        tr = np.maximum(
            high_vals - low_vals,
            np.maximum(
                np.abs(high_vals - np.roll(close.values, 1)),
                np.abs(low_vals - np.roll(close.values, 1)),
            ),
        )
        tr[0] = high_vals[0] - low_vals[0]
        atr = pd.Series(tr).rolling(vol_lookback, min_periods=1).mean().values
        atr = np.maximum(atr, 1e-10)

        drift_raw = np.abs(close.values - sma) / (atr * np.sqrt(vol_lookback))
        drift_scores = np.clip(drift_raw, 0.0, 1.0)

    # ── Per-bar systemic stress (realized vol percentile) ──
    # Uses a rolling percentile of realized volatility as a proxy for
    # market-wide stress.  Higher vol relative to history = higher stress.
    systemic_stress = np.zeros(n)
    if n >= vol_lookback:
        rolling_vol = pd.Series(returns).rolling(vol_lookback, min_periods=1).std().values
        rolling_vol = np.nan_to_num(rolling_vol, nan=0.0)
        # Vectorized expanding percentile: rank / count gives percentile in O(n)
        vol_series = pd.Series(rolling_vol)
        expanding_rank = vol_series.expanding(min_periods=2).rank()
        expanding_count = vol_series.expanding(min_periods=2).count()
        systemic_stress_series = (expanding_rank / expanding_count).fillna(0.0).clip(0.0, 1.0)

        # Zero out the first vol_lookback bars (no valid rolling vol)
        systemic_stress_series.iloc[:vol_lookback] = 0.0

        # Apply the max-check: if expanding max of rolling_vol is <= 1e-10, set to 0
        expanding_max = vol_series.expanding(min_periods=1).max()
        systemic_stress_series[expanding_max <= 1e-10] = 0.0

        systemic_stress = systemic_stress_series.values

    # ── Build ShockVectors ──
    shock_vectors: Dict = {}
    for i in range(n):
        idx = ohlcv.index[i]
        ts = idx if hasattr(idx, 'isoformat') else datetime.now()

        structural_features = _build_structural_features(
            drift_score=float(drift_scores[i]),
            systemic_stress=float(systemic_stress[i]),
            features=features,
            bar_idx=i,
        )

        sv = ShockVector(
            schema_version=SHOCK_VECTOR_SCHEMA_VERSION,
            timestamp=ts,
            ticker=ticker,
            hmm_regime=int(regime_vals[i]),
            hmm_confidence=float(conf_vals[i]),
            hmm_uncertainty=float(uncertainty_vals[i]),
            bocpd_changepoint_prob=float(bocpd_cp_probs[i]),
            bocpd_runlength=int(bocpd_runlengths[i]),
            jump_detected=bool(jump_flags[i]),
            jump_magnitude=float(jump_magnitudes[i]),
            structural_features=structural_features,
            transition_matrix=None,
            ensemble_model_type=ensemble_model_type,
        )
        shock_vectors[idx] = sv

    logger.debug(
        "compute_shock_vectors: computed %d vectors for %s", len(shock_vectors), ticker,
    )
    return shock_vectors
