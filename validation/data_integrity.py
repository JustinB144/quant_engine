"""
Data integrity preflight — Truth Layer T2.

Validates that all OHLCV data in a universe passes quality checks
before it enters the modeling or backtesting pipeline.  When
``fail_fast=True``, the first corrupted ticker raises immediately.

Usage:
    validator = DataIntegrityValidator(fail_fast=True)
    result = validator.validate_universe(ohlcv_dict)
    if not result.passed:
        raise RuntimeError(f"{result.n_stocks_failed} corrupted tickers")
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from ..data.quality import assess_ohlcv_quality, DataQualityReport

logger = logging.getLogger(__name__)


@dataclass
class DataIntegrityCheckResult:
    """Aggregate result of data integrity checks across a universe."""
    passed: bool
    n_stocks_passed: int
    n_stocks_failed: int
    failed_tickers: List[str]
    failed_reasons: Dict[str, List[str]] = field(default_factory=dict)
    reports: Dict[str, DataQualityReport] = field(default_factory=dict, repr=False)


class DataIntegrityValidator:
    """Preflight validator that blocks corrupt OHLCV from entering the pipeline.

    Parameters
    ----------
    fail_fast : bool
        When True (default), raise ``RuntimeError`` on the first corrupted
        ticker.  When False, check all tickers and return the aggregate result.
    """

    def __init__(self, fail_fast: bool = True):
        self.fail_fast = fail_fast

    def validate_universe(
        self,
        ohlcv_dict: Dict[str, pd.DataFrame],
    ) -> DataIntegrityCheckResult:
        """Check all stocks in a universe for data quality.

        Parameters
        ----------
        ohlcv_dict : dict[str, pd.DataFrame]
            Mapping of ticker/permno -> OHLCV DataFrame.

        Returns
        -------
        DataIntegrityCheckResult
            Aggregate pass/fail result with per-ticker detail.

        Raises
        ------
        RuntimeError
            If ``fail_fast=True`` and any ticker fails quality checks.
        """
        failed_tickers: List[str] = []
        failed_reasons: Dict[str, List[str]] = {}
        reports: Dict[str, DataQualityReport] = {}

        for ticker, df in ohlcv_dict.items():
            report = assess_ohlcv_quality(df, fail_on_error=False)
            reports[ticker] = report

            if not report.passed:
                failed_tickers.append(ticker)
                failed_reasons[ticker] = list(report.warnings)
                logger.warning(
                    "Data integrity FAILED for %s: %s",
                    ticker,
                    "; ".join(report.warnings),
                )
                if self.fail_fast:
                    raise RuntimeError(
                        f"Data integrity check failed for {ticker}: "
                        f"{'; '.join(report.warnings)}"
                    )

        n_passed = len(ohlcv_dict) - len(failed_tickers)
        result = DataIntegrityCheckResult(
            passed=len(failed_tickers) == 0,
            n_stocks_passed=n_passed,
            n_stocks_failed=len(failed_tickers),
            failed_tickers=failed_tickers,
            failed_reasons=failed_reasons,
            reports=reports,
        )

        if result.passed:
            logger.info(
                "Data integrity check PASSED: %d stocks validated", n_passed
            )
        else:
            logger.error(
                "Data integrity check FAILED: %d/%d stocks corrupted — %s",
                result.n_stocks_failed,
                len(ohlcv_dict),
                ", ".join(result.failed_tickers[:10]),
            )

        return result
