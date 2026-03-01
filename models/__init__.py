"""
Models subpackage — training, prediction, versioning, and retraining triggers.
"""

from .governance import ModelGovernance, ChampionRecord
from .cross_sectional import cross_sectional_rank
from .calibration import ConfidenceCalibrator
from .neural_net import TabularNet
from .walk_forward import walk_forward_select
from .feature_stability import FeatureStabilityTracker
from .shift_detection import DistributionShiftDetector
from .conformal import ConformalPredictor
from .trainer import ModelTrainer
from .predictor import EnsemblePredictor
from .versioning import ModelRegistry, ModelVersion
from .retrain_trigger import RetrainTrigger

__all__ = [
    "ModelGovernance",
    "ChampionRecord",
    "cross_sectional_rank",
    "ConfidenceCalibrator",
    "TabularNet",
    "walk_forward_select",
    "FeatureStabilityTracker",
    "DistributionShiftDetector",
    "ConformalPredictor",
    "ModelTrainer",
    "EnsemblePredictor",
    "ModelRegistry",
    "ModelVersion",
    "RetrainTrigger",
]
