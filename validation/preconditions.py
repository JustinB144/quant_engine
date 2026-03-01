"""
Execution contract validation — Truth Layer T1.

Validates that global preconditions (RET_TYPE, LABEL_H, PX_TYPE,
ENTRY_PRICE_TYPE) are locked down and sensible before any modeling
or backtesting begins.

Usage:
    ok, msg = validate_execution_contract()
    if not ok:
        raise RuntimeError(f"Preconditions failed: {msg}")
"""
import logging
from typing import Tuple

from ..config import (
    RET_TYPE,
    LABEL_H,
    PX_TYPE,
    ENTRY_PRICE_TYPE,
    TRUTH_LAYER_STRICT_PRECONDITIONS,
)
from ..config_structured import PreconditionsConfig

logger = logging.getLogger(__name__)


def validate_execution_contract() -> Tuple[bool, str]:
    """Check that execution preconditions are locked and sensible.

    Reads the four execution contract constants from ``config.py`` and
    validates them through the typed ``PreconditionsConfig`` dataclass.

    Returns
    -------
    tuple[bool, str]
        ``(True, summary_string)`` on success, ``(False, error_message)``
        on failure.
    """
    try:
        cfg = PreconditionsConfig(
            ret_type=RET_TYPE,
            label_h=LABEL_H,
            px_type=PX_TYPE,
            entry_price_type=ENTRY_PRICE_TYPE,
        )
        summary = (
            f"Execution contract OK: ret_type={cfg.ret_type.value}, "
            f"label_h={cfg.label_h}, px_type={cfg.px_type.value}, "
            f"entry_price_type={cfg.entry_price_type.value}"
        )
        logger.info(summary)
        return True, summary
    except (ValueError, KeyError) as e:
        error_msg = f"Execution contract validation failed: {e}"
        logger.error(error_msg)
        return False, error_msg


def enforce_preconditions() -> None:
    """Validate execution contract; raise RuntimeError on failure.

    Called from ``Backtester.__init__()`` and ``ModelTrainer.__init__()``
    when ``TRUTH_LAYER_STRICT_PRECONDITIONS`` is True.

    When strict mode is off, validation still runs but violations are
    logged as warnings instead of raising exceptions.
    """
    if not TRUTH_LAYER_STRICT_PRECONDITIONS:
        # Warn-only mode: still validate but log instead of raising
        ok, msg = validate_execution_contract()
        if not ok:
            logger.warning("Precondition violation (non-strict mode): %s", msg)
        return

    ok, msg = validate_execution_contract()
    if not ok:
        raise RuntimeError(msg)
