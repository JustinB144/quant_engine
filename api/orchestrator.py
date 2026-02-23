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
        from quant_engine.data.loader import load_survivorship_universe, load_universe
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
            # Build a diagnostic message so the user knows *why* data loading failed
            diagnostics = []
            ticker_list = tickers if tickers else ("survivorship universe",)
            diagnostics.append(f"Attempted tickers: {ticker_list}")
            diagnostics.append(f"WRDS_ENABLED={WRDS_ENABLED}")
            diagnostics.append(f"REQUIRE_PERMNO={REQUIRE_PERMNO}")
            diagnostics.append(f"years={years}")

            # Check if any cached data exists at all
            from quant_engine.config import DATA_DIR
            cache_dir = DATA_DIR / "cache"
            if cache_dir.exists():
                parquet_count = len(list(cache_dir.glob("*_1d.parquet")))
                daily_count = len(list(cache_dir.glob("*_daily_*.parquet")))
                diagnostics.append(f"Cache has {parquet_count} _1d.parquet + {daily_count} _daily_ files")
            else:
                diagnostics.append("Cache directory does not exist")

            detail = "; ".join(diagnostics)
            raise RuntimeError(
                f"No data loaded — all tickers were rejected or unavailable. {detail}"
            )

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
            except (ValueError, KeyError, TypeError):
                pass
            if progress_callback and i % 5 == 0:
                progress_callback(0.6 + 0.3 * (i / total), f"Predicting {i+1}/{total}")

        if not all_preds:
            return {"signals": [], "total": 0}

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
        }

    def backtest(
        self,
        state: PipelineState,
        horizon: int = 10,
        version: str = "latest",
        risk_management: bool = False,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run backtest on historical data."""
        import json as _json

        from quant_engine.backtest.engine import Backtester
        from quant_engine.config import ENTRY_THRESHOLD, REGIME_NAMES, RESULTS_DIR
        from quant_engine.models.predictor import EnsemblePredictor
        from quant_engine.regime.detector import RegimeDetector

        if progress_callback:
            progress_callback(0.6, "Loading model for backtest")
        predictor = EnsemblePredictor(horizon=horizon, version=version)
        detector = RegimeDetector()
        available_permnos = set(state.features.index.get_level_values(0))

        all_preds = []
        total = len(state.data)
        for i, (permno, _) in enumerate(state.data.items()):
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
            except (ValueError, KeyError, TypeError):
                pass
            if progress_callback and i % 5 == 0:
                progress_callback(0.6 + 0.2 * (i / total), f"Predicting {i+1}/{total}")

        if not all_preds:
            return {"error": "No predictions generated"}

        predictions = pd.concat(all_preds)

        if progress_callback:
            progress_callback(0.85, "Running backtest")
        backtester = Backtester(holding_days=horizon, use_risk_management=risk_management)
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
                })
            pd.DataFrame(trade_data).to_csv(RESULTS_DIR / f"backtest_{horizon}d_trades.csv", index=False)

            summary = {
                "horizon": horizon,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "avg_return": result.avg_return,
                "sharpe": result.sharpe_ratio,
                "sortino": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
                "annualized_return": result.annualized_return,
                "trades_per_year": result.trades_per_year,
                "regime_breakdown": result.regime_breakdown,
            }
            with open(RESULTS_DIR / f"backtest_{horizon}d_summary.json", "w") as f:
                _json.dump(summary, f, indent=2, default=str)

        if progress_callback:
            progress_callback(1.0, "Backtest complete")
        return {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
        }
