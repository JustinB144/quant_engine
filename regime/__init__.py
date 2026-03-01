"""Regime modeling components."""

# Core imports (no heavy dependencies)
from .confidence_calibrator import ConfidenceCalibrator
from .consensus import RegimeConsensus
from .correlation import CorrelationRegimeDetector
from .detector import (
    RegimeDetector,
    RegimeOutput,
    detect_regimes_batch,
    validate_hmm_observation_features,
)
from .hmm import GaussianHMM, HMMFitResult
from .jump_model import StatisticalJumpModel, JumpModelResult
from .jump_model_pypi import PyPIJumpModel
from .online_update import OnlineRegimeUpdater
from .shock_vector import ShockVector, ShockVectorValidator, compute_shock_vectors
from .uncertainty_gate import UncertaintyGate

# BOCPD requires scipy — import lazily to avoid hard scipy dependency
# for consumers that don't use change-point detection.
def __getattr__(name):
    """Lazy import for optional BOCPD components."""
    if name in ("BOCPDDetector", "BOCPDResult", "BOCPDBatchResult"):
        from .bocpd import BOCPDDetector, BOCPDResult, BOCPDBatchResult
        globals()["BOCPDDetector"] = BOCPDDetector
        globals()["BOCPDResult"] = BOCPDResult
        globals()["BOCPDBatchResult"] = BOCPDBatchResult
        return globals()[name]
    raise AttributeError(f"module 'regime' has no attribute {name!r}")

__all__ = [
    "BOCPDDetector",
    "BOCPDResult",
    "BOCPDBatchResult",
    "ConfidenceCalibrator",
    "CorrelationRegimeDetector",
    "RegimeConsensus",
    "RegimeDetector",
    "RegimeOutput",
    "GaussianHMM",
    "HMMFitResult",
    "OnlineRegimeUpdater",
    "StatisticalJumpModel",
    "JumpModelResult",
    "PyPIJumpModel",
    "ShockVector",
    "ShockVectorValidator",
    "UncertaintyGate",
    "compute_shock_vectors",
    "detect_regimes_batch",
    "validate_hmm_observation_features",
]
