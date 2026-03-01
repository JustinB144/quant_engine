"""Unified pipeline orchestrator — data -> features -> regimes -> compute.

Mirrors the flow in ``run_train.py``, ``run_predict.py``, ``run_backtest.py``
but returns structured dicts suitable for the API layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Intermediate state passed between orchestrator stages."""

    data: Dict[Any, pd.DataFrame] = field(default_factory=dict)
    features: Optional[pd.DataFrame] = None
    targets: Optional[pd.DataFrame] = None
    regimes: Optional[pd.Series] = None
    regime_probs: Optional[pd.DataFrame] = None


class PipelineOrchestrator:
    """Chains engine modules into a reproducible pipeline."""

    def load_and_prepare(
        self,
        tickers: Optional[List[str]] = None,
        years: int = 5,
        feature_mode: str = "core",
        survivorship_mode: bool = False,
        full_universe: bool = False,
        progress_callback=None,
    ) -> PipelineState:
        """Load data, compute features, detect regimes."""
        from quant_engine.config import UNIVERSE_FULL, UNIVERSE_QUICK, WRDS_ENABLED, REQUIRE_PERMNO
        from quant_engine.data.loader import load_survivorship_universe, load_universe, warn_if_survivorship_biased
        from quant_engine.features.pipeline import FeaturePipeline
        from quant_engine.regime.detector import RegimeDetector

        state = PipelineState()

        # Step 1: load data
        if progress_callback:
            progress_callback(0.1, "Loading data")
        if survivorship_mode:
            state.data = load_survivorship_universe(years=years, verbose=False)
        else:
            if tickers is None:
                tickers = UNIVERSE_FULL if full_universe else UNIVERSE_QUICK
            state.data = load_universe(tickers, years=years, verbose=False)

        if not state.data:
            from quant_engine.config import DATA_CACHE_DIR
            from quant_engine.data.loader import get_skip_reasons

            ticker_list = tickers if tickers else []
            n_tickers = len(ticker_list)
            skip_reasons = get_skip_reasons()

            # Cache diagnostics
            cache_dir = DATA_CACHE_DIR
            if cache_dir.exists():
                parquet_1d = len(list(cache_dir.glob("*_1d.parquet")))
                parquet_daily = len(list(cache_dir.glob("*_daily_*.parquet")))
                cache_info = f"{parquet_1d} _1d.parquet files, {parquet_daily} _daily_ files"
            else:
                cache_info = "directory does not exist"

            # Group skip reasons by category and add remediation
            reason_groups: Dict[str, List[str]] = {}
            for sym, reason in skip_reasons.items():
                reason_groups.setdefault(reason, []).append(sym)

            remediation_map = {
                "permno unresolved": "Set REQUIRE_PERMNO=False in config.py or run run_wrds_daily_refresh.py",
                "load_ohlcv returned None": "Check data sources and cache; run run_wrds_daily_refresh.py",
            }

            skip_lines = []
            for reason, syms in sorted(reason_groups.items(), key=lambda x: -len(x[1])):
                remedy = remediation_map.get(reason, "")
                if not remedy:
                    if "insufficient data" in reason:
                        remedy = "Increase years parameter or check data sources"
                    elif "quality" in reason:
                        remedy = "Check MAX_ZERO_VOLUME_FRACTION in config.py"
                    else:
                        remedy = "Check logs for per-ticker details"
                skip_lines.append(f"  - {len(syms)} tickers: \"{reason}\" -> {remedy}")

            msg_parts = [
                f"No data loaded — all {n_tickers} tickers were rejected.",
                "",
                "Diagnostics:",
                f"  WRDS_ENABLED={WRDS_ENABLED}, REQUIRE_PERMNO={REQUIRE_PERMNO}, years={years}",
                f"  Cache: {cache_info}",
            ]
            if skip_lines:
                msg_parts.append("")
                msg_parts.append("Top skip reasons:")
                msg_parts.extend(skip_lines)
            msg_parts.append("")
            msg_parts.append("To debug: Set verbose=True in load_universe() or check logs for per-ticker details.")

            raise RuntimeError("\n".join(msg_parts))

        warn_if_survivorship_biased(state.data, context="orchestrator pipeline")

        # Step 2: features
        if progress_callback:
            progress_callback(0.3, f"Computing features ({feature_mode})")
        pipeline = FeaturePipeline(
            feature_mode=feature_mode,
            include_interactions=True,
            verbose=False,
        )
        state.features, state.targets = pipeline.compute_universe(state.data, verbose=False)

        # Step 3: regimes
        if progress_callback:
            progress_callback(0.5, "Detecting regimes")
        detector = RegimeDetector()
        regime_dfs = []
        regime_prob_dfs = []
        for permno in state.data:
            if permno not in state.features.index.get_level_values(0):
                continue
            permno_feats = state.features.loc[permno]
            regime_df = detector.regime_features(permno_feats)
            regime_df["permno"] = permno
            regime_df = regime_df.set_index("permno", append=True).reorder_levels([1, 0])
            regime_dfs.append(regime_df)
            prob_cols = [c for c in regime_df.columns if c.startswith("regime_prob_")]
            if prob_cols:
                regime_prob_dfs.append(regime_df[prob_cols])

        if regime_dfs:
            all_regime = pd.concat(regime_dfs)
            state.regimes = all_regime["regime"]
            state.regime_probs = pd.concat(regime_prob_dfs) if regime_prob_dfs else None

        # Invalidate stale API caches after fresh data load
        try:
            from .cache.invalidation import invalidate_on_data_refresh
            from .deps.providers import get_cache
            invalidate_on_data_refresh(get_cache())
        except Exception:
            pass  # Cache invalidation failure is non-fatal

        if progress_callback:
            progress_callback(0.6, "Pipeline ready")
        return state

    def train(
        self,
        state: PipelineState,
        horizons: List[int],
        survivorship_mode: bool = False,
        recency_weight: bool = False,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Train ensemble models for the given horizons."""
        from quant_engine.models.governance import ModelGovernance
        from quant_engine.models.trainer import ModelTrainer
        from quant_engine.models.versioning import ModelRegistry

        trainer = ModelTrainer()
        governance = ModelGovernance()
        registry = ModelRegistry()
        results = {}

        for i, horizon in enumerate(horizons):
            target_col = f"target_{horizon}d"
            if state.targets is None or target_col not in state.targets.columns:
                results[str(horizon)] = {"error": f"{target_col} not in targets"}
                continue

            if progress_callback:
                frac = 0.6 + 0.3 * (i / max(len(horizons), 1))
                progress_callback(frac, f"Training {horizon}d model")

            result = trainer.train_ensemble(
                features=state.features,
                targets=state.targets[target_col],
                regimes=state.regimes,
                regime_probabilities=state.regime_probs,
                horizon=horizon,
                verbose=False,
                versioned=True,
                survivorship_mode=survivorship_mode,
                recency_weight=recency_weight,
            )

            if result.global_model is None:
                logger.warning(
                    "Training rejected by quality gates for horizon=%d — "
                    "no model was saved.",
                    horizon,
                )
                results[str(horizon)] = {
                    "status": "rejected",
                    "reason": "quality_gates",
                }
                continue

            latest_id = registry.latest_version_id
            gov_result = {}
            if latest_id is not None:
                gov_result = governance.evaluate_and_update(
                    horizon=horizon,
                    version_id=latest_id,
                    metrics={
                        "oos_spearman": float(np.mean(result.global_model.cv_scores) if result.global_model.cv_scores else 0),
                        "holdout_spearman": float(result.global_model.holdout_correlation),
                        "cv_gap": float(result.global_model.cv_gap),
                    },
                )

            results[str(horizon)] = {
                "version_id": latest_id,
                "governance": gov_result,
                "cv_gap": float(result.global_model.cv_gap),
                "holdout_correlation": float(result.global_model.holdout_correlation),
            }

        if progress_callback:
            progress_callback(1.0, "Training complete")
        return results

    def predict(
        self,
        state: PipelineState,
        horizon: int = 10,
        version: str = "latest",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Generate predictions using a trained model."""
        from quant_engine.config import ENTRY_THRESHOLD, CONFIDENCE_THRESHOLD, REGIME_NAMES, RESULTS_DIR
        from quant_engine.models.predictor import EnsemblePredictor
        from quant_engine.regime.detector import RegimeDetector

        if progress_callback:
            progress_callback(0.6, "Loading model")
        predictor = EnsemblePredictor(horizon=horizon, version=version)
        detector = RegimeDetector()
        available_permnos = set(state.features.index.get_level_values(0))

        all_preds = []
        failed_tickers: list = []
        total = len(state.data)
        for i, (permno, df) in enumerate(state.data.items()):
            if permno not in available_permnos:
                continue
            features = state.features.loc[permno]
            regime_df = detector.regime_features(features)
            regimes = regime_df["regime"]
            confidence = regime_df["regime_confidence"]
            regime_probs = regime_df[[c for c in regime_df.columns if c.startswith("regime_prob_")]]
            try:
                preds = predictor.predict(features, regimes, confidence, regime_probabilities=regime_probs)
                preds["permno"] = str(permno)
                preds["ticker"] = str(df.attrs.get("ticker", ""))
                all_preds.append(preds)
            except (ValueError, KeyError, TypeError) as exc:
                ticker = str(df.attrs.get("ticker", permno))
                logger.warning("Prediction failed for %s: %s", ticker, exc)
                failed_tickers.append({"ticker": ticker, "error": str(exc)})
            if progress_callback and i % 5 == 0:
                progress_callback(0.6 + 0.3 * (i / total), f"Predicting {i+1}/{total}")

        attempted = len(all_preds) + len(failed_tickers)
        if attempted > 0 and len(failed_tickers) / attempted > 0.5:
            logger.error(
                "Majority prediction failure: %d/%d tickers failed",
                len(failed_tickers), attempted,
            )

        if not all_preds:
            return {"signals": [], "total": 0, "failed_tickers": failed_tickers}

        predictions = pd.concat(all_preds, ignore_index=True)
        latest = predictions.groupby("permno").last().reset_index()
        latest = latest.sort_values("predicted_return", ascending=False)

        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"predictions_{horizon}d.csv"
        latest.to_csv(out_path, index=False)

        signals = latest[
            (latest["predicted_return"] > ENTRY_THRESHOLD)
            & (latest["confidence"] > CONFIDENCE_THRESHOLD)
        ]

        if progress_callback:
            progress_callback(1.0, "Predictions complete")
        return {
            "total_permnos": len(latest),
            "actionable_signals": len(signals),
            "output_path": str(out_path),
            "failed_tickers": failed_tickers,
        }

    def backtest(
        self,
        state: PipelineState,
        horizon: int = 10,
        version: str = "latest",
        risk_management: bool = False,
        holding_period: Optional[int] = None,
        max_positions: Optional[int] = None,
        entry_threshold: Optional[float] = None,
        position_size: Optional[float] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run backtest on historical data."""
        import json as _json

        from quant_engine.backtest.engine import Backtester, backtest_result_to_summary_dict
        from quant_engine.config import ENTRY_THRESHOLD, REGIME_NAMES, RESULTS_DIR
        from quant_engine.models.predictor import EnsemblePredictor
        from quant_engine.regime.detector import RegimeDetector

        if progress_callback:
            progress_callback(0.6, "Loading model for backtest")
        predictor = EnsemblePredictor(horizon=horizon, version=version)
        detector = RegimeDetector()
        available_permnos = set(state.features.index.get_level_values(0))

        all_preds = []
        failed_tickers: list = []
        total = len(state.data)
        for i, (permno, df_bt) in enumerate(state.data.items()):
            if permno not in available_permnos:
                continue
            features = state.features.loc[permno]
            regime_df = detector.regime_features(features)
            regimes = regime_df["regime"]
            confidence = regime_df["regime_confidence"]
            regime_probs = regime_df[[c for c in regime_df.columns if c.startswith("regime_prob_")]]
            try:
                preds = predictor.predict(features, regimes, confidence, regime_probabilities=regime_probs)
                preds["permno"] = permno
                preds = preds.set_index("permno", append=True).reorder_levels([1, 0])
                all_preds.append(preds)
            except (ValueError, KeyError, TypeError) as exc:
                ticker = str(df_bt.attrs.get("ticker", permno))
                logger.warning("Backtest prediction failed for %s: %s", ticker, exc)
                failed_tickers.append({"ticker": ticker, "error": str(exc)})
            if progress_callback and i % 5 == 0:
                progress_callback(0.6 + 0.2 * (i / total), f"Predicting {i+1}/{total}")

        attempted = len(all_preds) + len(failed_tickers)
        if attempted > 0 and len(failed_tickers) / attempted > 0.5:
            logger.error(
                "Majority backtest prediction failure: %d/%d tickers failed",
                len(failed_tickers), attempted,
            )

        if not all_preds:
            return {"error": "No predictions generated", "failed_tickers": failed_tickers}

        predictions = pd.concat(all_preds)

        if progress_callback:
            progress_callback(0.85, "Running backtest")
        bt_kwargs: Dict[str, Any] = {
            "holding_days": holding_period if holding_period is not None else horizon,
            "use_risk_management": risk_management,
        }
        if entry_threshold is not None:
            bt_kwargs["entry_threshold"] = entry_threshold
        if max_positions is not None:
            bt_kwargs["max_positions"] = max_positions
        if position_size is not None:
            bt_kwargs["position_size_pct"] = position_size
        backtester = Backtester(**bt_kwargs)
        result = backtester.run(predictions, state.data, verbose=False)

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if result.total_trades > 0:
            trade_data = []
            for t in result.trades:
                trade_data.append({
                    "permno": t.ticker,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "predicted_return": t.predicted_return,
                    "actual_return": t.actual_return,
                    "net_return": t.net_return,
                    "regime": REGIME_NAMES.get(t.regime, f"regime_{t.regime}"),
                    "confidence": t.confidence,
                    "holding_days": t.holding_days,
                    "exit_reason": t.exit_reason,
                    "position_size": t.position_size,
                })
            pd.DataFrame(trade_data).to_csv(RESULTS_DIR / f"backtest_{horizon}d_trades.csv", index=False)

            summary = backtest_result_to_summary_dict(result, horizon=horizon)
            with open(RESULTS_DIR / f"backtest_{horizon}d_summary.json", "w") as f:
                _json.dump(summary, f, indent=2, default=str)
        else:
            # Write zero-trade summary to prevent stale artifacts
            summary = {
                "horizon": horizon,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "annualized_return": 0.0,
                "trades_per_year": 0.0,
                "regime_breakdown": {},
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_return": 0.0,
                "avg_holding_days": 0.0,
            }
            with open(RESULTS_DIR / f"backtest_{horizon}d_summary.json", "w") as f:
                _json.dump(summary, f, indent=2, default=str)

            # Write empty trade CSV to prevent stale trade data
            pd.DataFrame().to_csv(RESULTS_DIR / f"backtest_{horizon}d_trades.csv", index=False)
            logger.info("Zero trades — wrote empty summary and cleared trade file.")

        if progress_callback:
            progress_callback(1.0, "Backtest complete")
        return {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "failed_tickers": failed_tickers,
        }
