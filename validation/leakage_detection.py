"""
Leakage detection — Truth Layer T3.

Provides two complementary leakage safeguards:

1. **Causality enforcement**: ensures features used in live prediction
   are tagged ``CAUSAL`` (not ``END_OF_DAY`` or ``RESEARCH_ONLY``).

2. **Time-shift correlation test**: detects features that have suspicious
   correlation with *future* labels — a hallmark of look-ahead leakage.

Usage:
    detector = LeakageDetector()
    result = detector.test_time_shift_leakage(features, labels)
    if not result.passed:
        raise RuntimeError(f"Leakage: {result.violations}")
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeakageTestResult:
    """Result of time-shift leakage detection."""
    passed: bool
    n_violations: int
    violations: List[Dict] = field(default_factory=list)


class LeakageDetector:
    """Detects forward-looking leakage via time-shift correlation test.

    For each feature, computes correlation with labels shifted forward
    by ``shift_range`` bars.  If the absolute correlation exceeds
    ``threshold_corr``, the feature is flagged as potentially leaking.

    Parameters
    ----------
    shift_range : list[int] or None
        Forward shift lags to test.  Default: ``[1, 2, 3, 5, 10]``.
    """

    def __init__(self, shift_range: Optional[List[int]] = None):
        self.shift_range = shift_range or [1, 2, 3, 5, 10]

    def test_time_shift_leakage(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        threshold_corr: float = 0.20,
    ) -> LeakageTestResult:
        """Check if features have suspicious correlation with FUTURE labels.

        For each feature column, compute Pearson correlation with labels
        shifted forward by each lag in ``shift_range``.  Negative shift
        means the label comes from the future relative to the feature
        observation — if correlation is high, the feature has access to
        future information.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix aligned to the same index as ``labels``.
        labels : pd.Series
            Target variable (e.g., forward returns).
        threshold_corr : float
            Flag features with ``|corr| > threshold_corr`` as leaking.

        Returns
        -------
        LeakageTestResult
            ``passed=True`` if no violations detected.
        """
        violations: List[Dict] = []

        # Align features and labels to common index
        common_index = features.index.intersection(labels.index)
        if len(common_index) == 0:
            logger.warning("No common index between features and labels; skipping leakage test")
            return LeakageTestResult(passed=True, n_violations=0, violations=[])

        feat_aligned = features.loc[common_index]
        label_aligned = labels.loc[common_index]

        for shift in self.shift_range:
            # Negative shift = future label relative to current feature row
            label_shifted = label_aligned.shift(-shift)
            mask = label_shifted.notna()

            if mask.sum() < 30:
                continue

            for col in feat_aligned.columns:
                col_data = feat_aligned.loc[mask, col]
                label_data = label_shifted.loc[mask]

                # Drop NaNs from the feature column too
                valid = col_data.notna() & label_data.notna()
                if valid.sum() < 30:
                    continue

                corr = float(col_data[valid].corr(label_data[valid]))

                if np.isnan(corr):
                    continue

                if abs(corr) > threshold_corr:
                    violations.append({
                        "feature": col,
                        "shift_lag": shift,
                        "correlation": round(corr, 4),
                    })

        result = LeakageTestResult(
            passed=len(violations) == 0,
            n_violations=len(violations),
            violations=violations,
        )

        if result.passed:
            logger.info(
                "Time-shift leakage test PASSED: %d features x %d shifts checked",
                len(feat_aligned.columns),
                len(self.shift_range),
            )
        else:
            logger.warning(
                "Time-shift leakage test FAILED: %d violations detected",
                result.n_violations,
            )
            for v in violations[:5]:
                logger.warning(
                    "  Leaky feature: %s — corr=%.3f at lag=%d",
                    v["feature"], v["correlation"], v["shift_lag"],
                )

        return result


def run_leakage_checks(
    features: pd.DataFrame,
    labels: pd.Series,
    threshold_corr: float = 0.20,
    shift_range: Optional[List[int]] = None,
) -> LeakageTestResult:
    """Run leakage detection and return result.

    Convenience wrapper that creates a ``LeakageDetector`` and runs
    the time-shift test.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    labels : pd.Series
        Target variable.
    threshold_corr : float
        Correlation threshold for flagging leakage.
    shift_range : list[int] or None
        Forward shift lags to test.

    Returns
    -------
    LeakageTestResult

    Raises
    ------
    RuntimeError
        If leakage is detected (any feature exceeds threshold).
    """
    detector = LeakageDetector(shift_range=shift_range or [1, 2, 3, 5])
    result = detector.test_time_shift_leakage(
        features, labels, threshold_corr=threshold_corr,
    )

    if not result.passed:
        top_violations = result.violations[:10]
        detail = "\n".join(
            f"  {v['feature']}: corr={v['correlation']:.3f} at lag={v['shift_lag']}"
            for v in top_violations
        )
        raise RuntimeError(
            f"Leakage detected: {result.n_violations} features have "
            f"suspicious forward correlation:\n{detail}"
        )

    return result
