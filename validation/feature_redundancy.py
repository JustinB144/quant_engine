"""
Feature Redundancy Detection — identifies highly correlated features.

Used to validate that structural features (spectral, SSA, tail risk,
eigenvalue, OT) do not introduce problematic redundancy with each other
or with existing classical features.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureRedundancyDetector:
    """Detect and report highly correlated features."""

    @staticmethod
    def detect_redundant_pairs(
        features: pd.DataFrame,
        threshold: float = 0.90,
    ) -> List[Tuple[str, str, float]]:
        """Find feature pairs with absolute correlation exceeding threshold.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (rows = observations, columns = features).
        threshold : float
            Minimum absolute correlation to flag as redundant.

        Returns
        -------
        List[Tuple[str, str, float]]
            Sorted list of (feature1, feature2, correlation) tuples,
            highest correlation first.
        """
        # Drop columns that are all NaN to avoid spurious correlations
        valid_features = features.dropna(axis=1, how="all")
        if valid_features.shape[1] < 2:
            return []

        corr = valid_features.corr().abs()
        pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_ij = corr.iloc[i, j]
                if np.isfinite(corr_ij) and corr_ij > threshold:
                    pairs.append((
                        corr.columns[i],
                        corr.columns[j],
                        float(corr_ij),
                    ))

        return sorted(pairs, key=lambda x: x[2], reverse=True)

    @staticmethod
    def report(redundancies: List[Tuple[str, str, float]]) -> str:
        """Generate human-readable redundancy report."""
        if not redundancies:
            return "No redundant features detected."

        lines = [f"Found {len(redundancies)} redundant pair(s):"]
        for feat1, feat2, corr in redundancies[:20]:
            lines.append(f"  {feat1} <-> {feat2}: {corr:.3f}")

        return "\n".join(lines)


def validate_structural_feature_composition(
    features: pd.DataFrame,
) -> bool:
    """Check that structural features don't introduce problematic redundancy.

    Specific checks:
        - SpectralEntropy vs SSASingularEnt: should be < 0.85 correlated
        - JumpIntensity vs ExtremeRetPct: should be < 0.80 correlated

    Parameters
    ----------
    features : pd.DataFrame
        Full feature matrix including structural features.

    Returns
    -------
    bool
        True if composition is acceptable, False if issues found.
    """
    structural_checks = [
        ("SpectralEntropy_252", "SSASingularEnt_60", 0.85),
        ("JumpIntensity_20", "ExtremeRetPct_20", 0.80),
    ]

    issues = []
    for feat1, feat2, max_corr in structural_checks:
        if feat1 in features.columns and feat2 in features.columns:
            valid = features[[feat1, feat2]].dropna()
            if len(valid) > 20:
                corr = valid[feat1].corr(valid[feat2])
                if np.isfinite(corr) and abs(corr) > max_corr:
                    issues.append(
                        f"High correlation between {feat1} and {feat2}: "
                        f"{corr:.3f} (threshold: {max_corr})"
                    )

    if issues:
        logger.warning("Feature composition issues:\n" + "\n".join(issues))
        return False

    return True
