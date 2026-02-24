"""
Backward-compatible re-export of legacy Statistical Jump Model.

The original implementation has been moved to ``jump_model_legacy.py``.
The production replacement is in ``jump_model_pypi.py`` (PyPI jumpmodels package).
"""
from .jump_model_legacy import JumpModelResult, StatisticalJumpModel

__all__ = ["JumpModelResult", "StatisticalJumpModel"]
