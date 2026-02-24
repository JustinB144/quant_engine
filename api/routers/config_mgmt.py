"""Runtime config management endpoints."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse

from ..cache.invalidation import invalidate_on_config_change
from ..cache.manager import CacheManager
from ..config import RuntimeConfig
from ..deps.providers import get_cache, get_runtime_config
from ..schemas.envelope import ApiResponse

router = APIRouter(prefix="/api/config", tags=["config"])


def _annotated(value: Any, status: str, reason: str | None = None) -> Dict[str, Any]:
    """Build a single config entry with status annotation."""
    entry: Dict[str, Any] = {"value": value, "status": status}
    if reason:
        entry["reason"] = reason
    return entry


def _build_config_status() -> Dict[str, Dict[str, Any]]:
    """Build the full annotated config status response.

    Reads live values from quant_engine.config and determines
    the effective status of each feature flag.
    """
    import quant_engine.config as cfg

    gics_empty = not cfg.GICS_SECTORS

    return {
        "regime": {
            "model_type": _annotated(cfg.REGIME_MODEL_TYPE, "active"),
            "ensemble_enabled": _annotated(cfg.REGIME_ENSEMBLE_ENABLED, "active"),
            "jump_model_enabled": _annotated(cfg.REGIME_JUMP_MODEL_ENABLED, "active"),
            "hmm_states": _annotated(cfg.REGIME_HMM_STATES, "active"),
            "hmm_auto_select_states": _annotated(cfg.REGIME_HMM_AUTO_SELECT_STATES, "active"),
            "regime_2_trade_enabled": _annotated(cfg.REGIME_2_TRADE_ENABLED, "active"),
        },
        "risk": {
            "max_sector_exposure": _annotated(
                cfg.MAX_SECTOR_EXPOSURE,
                "inactive" if gics_empty else "active",
                reason="GICS_SECTORS is empty" if gics_empty else None,
            ),
            "gics_sectors_populated": _annotated(not gics_empty, "active" if not gics_empty else "inactive"),
            "almgren_chriss_enabled": _annotated(cfg.ALMGREN_CHRISS_ENABLED, "placeholder"),
            "kelly_fraction": _annotated(cfg.KELLY_FRACTION, "active"),
            "kelly_bayesian_alpha": _annotated(cfg.KELLY_BAYESIAN_ALPHA, "placeholder"),
            "kelly_bayesian_beta": _annotated(cfg.KELLY_BAYESIAN_BETA, "placeholder"),
            "kelly_portfolio_blend": _annotated(cfg.KELLY_PORTFOLIO_BLEND, "placeholder"),
            "kelly_regime_conditional": _annotated(cfg.KELLY_REGIME_CONDITIONAL, "placeholder"),
            "max_portfolio_dd": _annotated(cfg.MAX_PORTFOLIO_DD, "placeholder"),
            "max_portfolio_vol": _annotated(cfg.MAX_PORTFOLIO_VOL, "active"),
        },
        "data": {
            "wrds_enabled": _annotated(cfg.WRDS_ENABLED, "active"),
            "optionmetrics_enabled": _annotated(cfg.OPTIONMETRICS_ENABLED, "placeholder"),
            "kalshi_enabled": _annotated(cfg.KALSHI_ENABLED, "active"),
            "data_quality_enabled": _annotated(cfg.DATA_QUALITY_ENABLED, "active"),
        },
        "model": {
            "ensemble_diversify": _annotated(cfg.ENSEMBLE_DIVERSIFY, "active"),
            "cv_folds": _annotated(cfg.CV_FOLDS, "active"),
            "holdout_fraction": _annotated(cfg.HOLDOUT_FRACTION, "active"),
            "max_features_selected": _annotated(cfg.MAX_FEATURES_SELECTED, "active"),
            "model_registry": _annotated(str(cfg.MODEL_REGISTRY), "placeholder"),
            "champion_registry": _annotated(str(cfg.CHAMPION_REGISTRY), "active"),
        },
        "backtest": {
            "transaction_cost_bps": _annotated(cfg.TRANSACTION_COST_BPS, "active"),
            "entry_threshold": _annotated(cfg.ENTRY_THRESHOLD, "active"),
            "confidence_threshold": _annotated(cfg.CONFIDENCE_THRESHOLD, "active"),
            "max_positions": _annotated(cfg.MAX_POSITIONS, "active"),
            "exec_dynamic_costs": _annotated(cfg.EXEC_DYNAMIC_COSTS, "active"),
            "wf_max_train_dates": _annotated(cfg.WF_MAX_TRAIN_DATES, "active"),
        },
        "retraining": {
            "retrain_max_days": _annotated(cfg.RETRAIN_MAX_DAYS, "active"),
            "retrain_min_trades": _annotated(cfg.RETRAIN_MIN_TRADES, "placeholder"),
            "retrain_min_win_rate": _annotated(cfg.RETRAIN_MIN_WIN_RATE, "placeholder"),
            "retrain_min_correlation": _annotated(cfg.RETRAIN_MIN_CORRELATION, "placeholder"),
        },
        "training": {
            "forward_horizons": _annotated(cfg.FORWARD_HORIZONS, "active"),
            "feature_mode": _annotated(cfg.AUTOPILOT_FEATURE_MODE, "active"),
        },
        "autopilot": {
            "feature_mode": _annotated(cfg.AUTOPILOT_FEATURE_MODE, "active"),
            "promotion_min_sharpe": _annotated(cfg.PROMOTION_MIN_SHARPE, "active"),
            "promotion_min_trades": _annotated(cfg.PROMOTION_MIN_TRADES, "active"),
            "paper_use_kelly_sizing": _annotated(cfg.PAPER_USE_KELLY_SIZING, "active"),
        },
    }


@router.get("")
async def get_config(rc: RuntimeConfig = Depends(get_runtime_config)) -> ApiResponse:
    return ApiResponse.success(rc.get_adjustable())


@router.get("/validate")
async def validate_config_endpoint() -> ApiResponse:
    """Run config validation and return any issues found.

    Each issue has a ``level`` (WARNING or ERROR) and a ``message``
    describing what is wrong and how to fix it.
    """
    from quant_engine.config import validate_config
    issues = validate_config()
    return ApiResponse.success({
        "issues": issues,
        "count": len(issues),
        "errors": sum(1 for i in issues if i.get("level") == "ERROR"),
        "warnings": sum(1 for i in issues if i.get("level") == "WARNING"),
    })


@router.get("/status")
async def get_config_status() -> ApiResponse:
    """Return all config values with active/placeholder/inactive status annotations.

    This endpoint helps the UI show which features are actually running
    vs which are defined but not yet wired end-to-end.
    """
    return ApiResponse.success(_build_config_status())


@router.patch("")
async def patch_config(
    updates: dict = Body(...),
    rc: RuntimeConfig = Depends(get_runtime_config),
    cache: CacheManager = Depends(get_cache),
) -> ApiResponse:
    try:
        new_state = rc.patch(updates)
    except (KeyError, ValueError) as exc:
        resp = ApiResponse.fail(str(exc))
        return JSONResponse(status_code=422, content=resp.model_dump())
    invalidate_on_config_change(cache)
    return ApiResponse.success(new_state)
